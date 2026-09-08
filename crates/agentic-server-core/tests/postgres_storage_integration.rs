use std::sync::Arc;
use std::time::Duration;

use agentic_core::config::{PostgresConfig, SqliteConfig};
use agentic_core::storage::{
    ConversationStore, InOutItem, ResponseMetadata, ResponseStore, StorageError, create_pool_with_configs,
    create_pool_with_schema_and_configs,
};
use agentic_core::types::io::{InputItem, InputMessage, InputMessageContent};
use tokio::sync::Barrier;
use tokio::time::timeout;

fn input_item(text: &str) -> InOutItem {
    InOutItem::Input(InputItem::Message(InputMessage {
        id: None,
        role: "user".to_owned(),
        status: None,
        content: InputMessageContent::Text(text.to_owned()),
    }))
}

#[tokio::test]
#[ignore = "requires TEST_POSTGRES_URL pointing to an isolated PostgreSQL database"]
#[allow(
    clippy::too_many_lines,
    reason = "keeps the complete index migration lifecycle in one integration test"
)]
async fn postgres_index_upgrade_preserves_existing_state() {
    let database_url = std::env::var("TEST_POSTGRES_URL").expect("TEST_POSTGRES_URL must be set");
    let pool = create_pool_with_configs(Some(&database_url), SqliteConfig::default(), PostgresConfig::default())
        .await
        .expect("create PostgreSQL upgrade-test pool");
    let schema = format!("upgrade_{}", uuid::Uuid::now_v7().simple());
    let mut connection = pool.acquire().await.expect("acquire PostgreSQL upgrade connection");
    sqlx::query(&format!("CREATE SCHEMA {schema}"))
        .execute(&mut *connection)
        .await
        .expect("create isolated upgrade schema");
    sqlx::query(&format!("SET search_path TO {schema}"))
        .execute(&mut *connection)
        .await
        .expect("select isolated upgrade schema");
    sqlx::raw_sql(include_str!("../migrations/0001_initial.sql"))
        .execute(&mut *connection)
        .await
        .expect("apply initial PostgreSQL schema");
    sqlx::raw_sql(include_str!("../migrations/0002_add_placeholders.sql"))
        .execute(&mut *connection)
        .await
        .expect("apply placeholder PostgreSQL schema");
    sqlx::query("INSERT INTO conversations (id, created_at, metadata) VALUES ($1, $2, $3)")
        .bind("conv_upgrade")
        .bind(1_704_067_200_i64)
        .bind("{\"source\":\"upgrade\"}")
        .execute(&mut *connection)
        .await
        .expect("seed conversation");
    sqlx::query("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES ($1, $2, $3, $4, $5)")
        .bind("item_upgrade")
        .bind("{}")
        .bind(1_704_067_200_i64)
        .bind("conv_upgrade")
        .bind(0_i64)
        .execute(&mut *connection)
        .await
        .expect("seed item");
    sqlx::query(
        "INSERT INTO responses \
         (id, conversation_id, history_item_ids, metadata, created_at) VALUES ($1, $2, $3, $4, $5)",
    )
    .bind("resp_upgrade")
    .bind("conv_upgrade")
    .bind("[\"item_upgrade\"]")
    .bind("{\"source\":\"upgrade\"}")
    .bind(1_704_067_200_i64)
    .execute(&mut *connection)
    .await
    .expect("seed response");

    sqlx::raw_sql(include_str!("../migrations/0003_index_conversation_sequence.sql"))
        .execute(&mut *connection)
        .await
        .expect("apply PostgreSQL composite index migration");

    let linked_state: (i64, String, String) = sqlx::query_as(
        "SELECT conversations.created_at, items.id, responses.id \
         FROM conversations \
         JOIN items ON items.conversation_id = conversations.id \
         JOIN responses ON responses.conversation_id = conversations.id \
         WHERE conversations.id = $1",
    )
    .bind("conv_upgrade")
    .fetch_one(&mut *connection)
    .await
    .expect("load migrated linked state");
    assert_eq!(
        linked_state,
        (1_704_067_200, "item_upgrade".to_owned(), "resp_upgrade".to_owned())
    );
    let recreated_indexes: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM pg_indexes \
         WHERE schemaname = $1 AND indexname IN ( \
           'idx_conversations_tenant_id', 'idx_items_conversation_id', \
           'idx_items_created_at', 'idx_items_tenant_id', \
           'idx_responses_conversation_id', 'idx_responses_previous_response_id', \
           'idx_responses_created_at', 'idx_responses_tenant_id' \
         )",
    )
    .bind(&schema)
    .fetch_one(&mut *connection)
    .await
    .expect("inspect recreated PostgreSQL indexes");
    assert_eq!(recreated_indexes, 8);
    let conversation_index: String = sqlx::query_scalar(
        "SELECT indexdef FROM pg_indexes \
         WHERE schemaname = $1 AND indexname = 'idx_items_conversation_id'",
    )
    .bind(&schema)
    .fetch_one(&mut *connection)
    .await
    .expect("inspect conversation sequence index");
    assert!(conversation_index.contains("(conversation_id, seq)"));
    let duplicate_sequence_error =
        sqlx::query("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES ($1, $2, $3, $4, $5)")
            .bind("item_duplicate_sequence")
            .bind("{}")
            .bind(1_704_067_200_i64)
            .bind("conv_upgrade")
            .bind(0_i64)
            .execute(&mut *connection)
            .await;
    assert!(
        duplicate_sequence_error.is_err(),
        "conversation sequences must be database-unique"
    );
    let foreign_key_error =
        sqlx::query("INSERT INTO items (id, data, created_at, conversation_id) VALUES ($1, $2, $3, $4)")
            .bind("item_invalid")
            .bind("{}")
            .bind(1_704_067_200_i64)
            .bind("conv_missing")
            .execute(&mut *connection)
            .await;
    assert!(foreign_key_error.is_err());

    sqlx::query("SET search_path TO public")
        .execute(&mut *connection)
        .await
        .expect("restore public schema");
    sqlx::query(&format!("DROP SCHEMA {schema} CASCADE"))
        .execute(&mut *connection)
        .await
        .expect("drop isolated upgrade schema");
    drop(connection);
    pool.close().await;
}

#[tokio::test]
#[ignore = "requires TEST_POSTGRES_URL pointing to an isolated PostgreSQL database"]
#[allow(
    clippy::too_many_lines,
    reason = "keeps the complete restart persistence lifecycle in one integration test"
)]
async fn postgres_migrations_and_state_survive_pool_restarts() {
    let database_url = std::env::var("TEST_POSTGRES_URL").expect("TEST_POSTGRES_URL must be set");
    let postgres_config = PostgresConfig {
        max_connections: 2,
        acquire_timeout: Duration::from_secs(5),
        lock_timeout: Duration::from_secs(1),
        migration_timeout: Duration::from_secs(5),
        statement_timeout: Duration::from_secs(5),
        idle_timeout: Some(Duration::from_secs(30)),
        max_lifetime: Some(Duration::from_secs(60)),
    };
    let response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
    let continuation_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
    let conversation_response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
    let conversation_continuation_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
    let metadata = ResponseMetadata::default();

    let first_pool = create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("initialize clean PostgreSQL database");
    assert_eq!(first_pool.options().get_max_connections(), 2);
    let statement_timeout: String = sqlx::query_scalar("SHOW statement_timeout")
        .fetch_one(first_pool.as_ref())
        .await
        .expect("inspect PostgreSQL statement timeout");
    assert_eq!(statement_timeout, "5s");
    let created_at_type: String = sqlx::query_scalar(
        "SELECT data_type FROM information_schema.columns \
         WHERE table_schema = current_schema() AND table_name = 'conversations' AND column_name = 'created_at'",
    )
    .fetch_one(first_pool.as_ref())
    .await
    .expect("inspect PostgreSQL timestamp column");
    assert_eq!(created_at_type, "bigint");
    let future_conversation_id = format!("conv_postgres_{}", uuid::Uuid::now_v7());
    let future_timestamp = i64::from(i32::MAX) + 1;
    sqlx::query("INSERT INTO conversations (id, created_at) VALUES ($1, $2)")
        .bind(&future_conversation_id)
        .bind(future_timestamp)
        .execute(first_pool.as_ref())
        .await
        .expect("persist timestamp beyond PostgreSQL INT4 range");
    let stored_future_timestamp: i64 = sqlx::query_scalar("SELECT created_at FROM conversations WHERE id = $1")
        .bind(&future_conversation_id)
        .fetch_one(first_pool.as_ref())
        .await
        .expect("load timestamp beyond PostgreSQL INT4 range");
    assert_eq!(stored_future_timestamp, future_timestamp);

    ResponseStore::new(first_pool.clone())
        .persist(&response_id, None, vec![input_item("first turn")], &metadata)
        .await
        .expect("persist first response");
    let first_conversation_store = ConversationStore::new(first_pool.clone());
    let conversation = first_conversation_store.create().await.expect("create conversation");
    first_conversation_store
        .persist(
            &conversation.conversation_id,
            &conversation_response_id,
            None,
            vec![
                input_item("conversation turn one"),
                input_item("conversation turn one follow-up"),
            ],
            &metadata,
        )
        .await
        .expect("persist first conversation turn");
    first_pool.close().await;

    let second_pool =
        create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
            .await
            .expect("repeat migrations after first restart");
    let second_store = ResponseStore::new(second_pool.clone());
    assert_eq!(
        second_store
            .rehydrate(&response_id)
            .await
            .expect("rehydrate first response after restart")
            .len(),
        1
    );
    second_store
        .persist(
            &continuation_id,
            Some(&response_id),
            vec![input_item("second turn")],
            &metadata,
        )
        .await
        .expect("persist continuation");
    let second_conversation_store = ConversationStore::new(second_pool.clone());
    assert_eq!(
        second_conversation_store
            .rehydrate(&conversation.conversation_id)
            .await
            .expect("rehydrate conversation after restart")
            .len(),
        2
    );
    second_conversation_store
        .persist(
            &conversation.conversation_id,
            &conversation_continuation_id,
            Some(&conversation_response_id),
            vec![input_item("conversation turn two")],
            &metadata,
        )
        .await
        .expect("persist second conversation turn");
    second_pool.close().await;

    let third_pool = create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("repeat migrations after second restart");
    assert_eq!(
        ResponseStore::new(third_pool.clone())
            .rehydrate(&continuation_id)
            .await
            .expect("rehydrate continuation after restart")
            .len(),
        2
    );
    assert_eq!(
        ConversationStore::new(third_pool)
            .rehydrate(&conversation.conversation_id)
            .await
            .expect("rehydrate full conversation after restart")
            .len(),
        3
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires TEST_POSTGRES_URL pointing to an isolated PostgreSQL database"]
async fn postgres_concurrent_conversation_writes_have_contiguous_sequences() {
    const WRITE_COUNT: usize = 12;

    let database_url = std::env::var("TEST_POSTGRES_URL").expect("TEST_POSTGRES_URL must be set");
    let postgres_config = PostgresConfig {
        max_connections: u32::try_from(WRITE_COUNT).expect("test write count must fit in u32"),
        acquire_timeout: Duration::from_secs(5),
        lock_timeout: Duration::from_secs(1),
        migration_timeout: Duration::from_secs(5),
        statement_timeout: Duration::from_secs(5),
        idle_timeout: Some(Duration::from_secs(30)),
        max_lifetime: Some(Duration::from_secs(60)),
    };
    let first_pool = create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("initialize PostgreSQL database");
    let second_pool = create_pool_with_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("create independent PostgreSQL pool");
    let conversation_store = ConversationStore::new(first_pool.clone());
    let conversation = conversation_store.create().await.expect("create conversation");
    let barrier = Arc::new(Barrier::new(WRITE_COUNT + 1));
    let mut tasks = Vec::with_capacity(WRITE_COUNT);

    for index in 0..WRITE_COUNT {
        let barrier = Arc::clone(&barrier);
        let pool = if index % 2 == 0 {
            first_pool.clone()
        } else {
            second_pool.clone()
        };
        let conversation_id = conversation.conversation_id.clone();
        tasks.push(tokio::spawn(async move {
            let store = ConversationStore::new(pool);
            let response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
            barrier.wait().await;
            store
                .persist(
                    &conversation_id,
                    &response_id,
                    None,
                    vec![input_item(&format!("concurrent turn {index}"))],
                    &ResponseMetadata::default(),
                )
                .await
        }));
    }

    barrier.wait().await;
    for task in tasks {
        task.await
            .expect("join concurrent write")
            .expect("persist concurrent turn");
    }

    let rows =
        agentic_core::storage::models::item::get_items_by_conversation(&first_pool, &conversation.conversation_id)
            .await
            .expect("load concurrent conversation items");
    let sequences = rows
        .iter()
        .map(|row| row.seq.expect("conversation item sequence"))
        .collect::<Vec<_>>();
    let write_count = i64::try_from(WRITE_COUNT).expect("test write count must fit in i64");
    assert_eq!(sequences, (0..write_count).collect::<Vec<_>>());

    first_pool.close().await;
    second_pool.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires TEST_POSTGRES_URL pointing to an isolated PostgreSQL database"]
#[allow(
    clippy::too_many_lines,
    reason = "keeps the complete two-pool race and persistence assertions in one integration test"
)]
async fn postgres_optimistic_conversation_conflict() {
    let database_url = std::env::var("TEST_POSTGRES_URL").expect("TEST_POSTGRES_URL must be set");
    let postgres_config = PostgresConfig {
        max_connections: 2,
        acquire_timeout: Duration::from_secs(5),
        lock_timeout: Duration::from_secs(1),
        migration_timeout: Duration::from_secs(5),
        statement_timeout: Duration::from_secs(5),
        idle_timeout: Some(Duration::from_secs(30)),
        max_lifetime: Some(Duration::from_secs(60)),
    };
    let first_pool = create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("initialize PostgreSQL database");
    let second_pool = create_pool_with_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("create independent PostgreSQL pool");
    let store = ConversationStore::new(first_pool.clone());
    let conversation = store.create().await.expect("create conversation");
    let version = store
        .rehydrate_snapshot(&conversation.conversation_id)
        .await
        .expect("capture conversation version")
        .version;
    let barrier = Arc::new(Barrier::new(2));

    let writer_one = {
        let store = ConversationStore::new(first_pool.clone());
        let conversation_id = conversation.conversation_id.clone();
        let barrier = Arc::clone(&barrier);
        let version = version.clone();
        tokio::spawn(async move {
            let response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
            let items = vec![input_item("writer one input"), input_item("writer one follow-up")];
            barrier.wait().await;
            let result = store
                .persist_if_version(
                    &conversation_id,
                    version,
                    &response_id,
                    None,
                    items.clone(),
                    &ResponseMetadata::default(),
                )
                .await;
            (result, response_id, items)
        })
    };
    let writer_two = {
        let store = ConversationStore::new(second_pool.clone());
        let conversation_id = conversation.conversation_id.clone();
        let barrier = Arc::clone(&barrier);
        tokio::spawn(async move {
            let response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
            let items = vec![input_item("writer two input"), input_item("writer two follow-up")];
            barrier.wait().await;
            let result = store
                .persist_if_version(
                    &conversation_id,
                    version,
                    &response_id,
                    None,
                    items.clone(),
                    &ResponseMetadata::default(),
                )
                .await;
            (result, response_id, items)
        })
    };

    let (writer_one_result, writer_one_response_id, writer_one_items) =
        writer_one.await.expect("join first checked write");
    let (writer_two_result, writer_two_response_id, writer_two_items) =
        writer_two.await.expect("join second checked write");

    assert_eq!(
        usize::from(writer_one_result.is_ok()) + usize::from(writer_two_result.is_ok()),
        1
    );
    assert_eq!(
        usize::from(matches!(
            &writer_one_result,
            Err(StorageError::ConversationConflict { conversation_id })
                if conversation_id == conversation.conversation_id.as_str()
        )) + usize::from(matches!(
            &writer_two_result,
            Err(StorageError::ConversationConflict { conversation_id })
                if conversation_id == conversation.conversation_id.as_str()
        )),
        1
    );
    let (winner_items, losing_response_id) = if writer_one_result.is_ok() {
        (writer_one_items, writer_two_response_id)
    } else {
        (writer_two_items, writer_one_response_id)
    };
    let rows =
        agentic_core::storage::models::item::get_items_by_conversation(&first_pool, &conversation.conversation_id)
            .await
            .expect("load winning conversation items");
    let sequences = rows
        .iter()
        .map(|row| row.seq.expect("conversation item sequence"))
        .collect::<Vec<_>>();
    assert_eq!(sequences, vec![0, 1]);
    assert_eq!(
        store
            .rehydrate(&conversation.conversation_id)
            .await
            .expect("rehydrate winning conversation items"),
        winner_items
    );
    let response_error = ResponseStore::new(first_pool.clone())
        .get(&losing_response_id)
        .await
        .expect_err("the losing response must not be stored");
    assert!(response_error.is_not_found());

    first_pool.close().await;
    second_pool.close().await;
}

#[tokio::test]
#[ignore = "requires TEST_POSTGRES_URL pointing to an isolated PostgreSQL database"]
async fn postgres_lock_wait_is_bounded_without_blocking_other_conversations() {
    let database_url = std::env::var("TEST_POSTGRES_URL").expect("TEST_POSTGRES_URL must be set");
    let postgres_config = PostgresConfig {
        max_connections: 3,
        acquire_timeout: Duration::from_secs(5),
        lock_timeout: Duration::from_secs(1),
        migration_timeout: Duration::from_secs(5),
        statement_timeout: Duration::from_secs(5),
        idle_timeout: Some(Duration::from_secs(30)),
        max_lifetime: Some(Duration::from_secs(60)),
    };
    let first_pool = create_pool_with_schema_and_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("initialize PostgreSQL database");
    let second_pool = create_pool_with_configs(Some(&database_url), SqliteConfig::default(), postgres_config)
        .await
        .expect("create independent PostgreSQL pool");
    let first_store = ConversationStore::new(first_pool.clone());
    let locked_conversation = first_store.create().await.expect("create locked conversation");
    let unrelated_conversation = first_store.create().await.expect("create unrelated conversation");
    let mut holding_transaction = first_pool.begin().await.expect("begin lock-holding transaction");
    agentic_core::storage::models::conversation::lock_in_tx(
        &mut holding_transaction,
        &locked_conversation.conversation_id,
    )
    .await
    .expect("hold conversation row lock");

    let blocked_store = ConversationStore::new(second_pool.clone());
    let blocked_conversation_id = locked_conversation.conversation_id.clone();
    let mut blocked_write = tokio::spawn(async move {
        blocked_store
            .persist(
                &blocked_conversation_id,
                &format!("resp_postgres_{}", uuid::Uuid::now_v7()),
                None,
                vec![input_item("blocked turn")],
                &ResponseMetadata::default(),
            )
            .await
    });
    let unrelated_store = ConversationStore::new(second_pool.clone());
    let unrelated_response_id = format!("resp_postgres_{}", uuid::Uuid::now_v7());
    let unrelated_metadata = ResponseMetadata::default();
    let unrelated_write = unrelated_store.persist(
        &unrelated_conversation.conversation_id,
        &unrelated_response_id,
        None,
        vec![input_item("unrelated turn")],
        &unrelated_metadata,
    );
    tokio::pin!(unrelated_write);
    tokio::select! {
        result = &mut unrelated_write => {
            result.expect("persist unrelated conversation turn");
        }
        result = &mut blocked_write => {
            let result = result.expect("join blocked write");
            panic!("locked write completed before unrelated conversation write: {result:?}");
        }
    }

    let blocked_result = timeout(Duration::from_secs(2), blocked_write)
        .await
        .expect("locked write should respect PostgreSQL lock_timeout")
        .expect("join blocked write");
    assert!(blocked_result.is_err(), "locked write unexpectedly succeeded");

    holding_transaction
        .rollback()
        .await
        .expect("release conversation row lock");
    first_pool.close().await;
    second_pool.close().await;
}
