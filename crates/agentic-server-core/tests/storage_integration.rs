mod support;

use agentic_core::config::SqliteConfig;
use agentic_core::storage::ResponseMetadata;
use agentic_core::storage::{
    ConversationStore, ResponseStore, create_pool_with_schema, create_pool_with_schema_and_sqlite_config,
};
use agentic_core::storage::{ConversationVersion, InOutItem, StorageError};
use agentic_core::types::event::MessageStatus;
use agentic_core::types::io::{InputItem, InputMessage, InputMessageContent, OutputItem, OutputMessage};
use std::sync::Arc;

use support::setup_pool;

fn create_input_item(text: &str) -> InOutItem {
    InOutItem::Input(InputItem::Message(InputMessage {
        id: None,
        role: "user".to_string(),
        status: None,
        content: InputMessageContent::Text(text.to_string()),
    }))
}

fn create_output_item(id: &str) -> InOutItem {
    InOutItem::Output(OutputItem::Message(OutputMessage::new(id, MessageStatus::Completed)))
}

#[tokio::test]
async fn test_conversation_store_create_and_get() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let created = store.create().await.expect("create failed");
    assert!(created.conversation_id.starts_with("conv_"));

    let retrieved = store.get(&created.conversation_id).await.expect("get failed");

    assert_eq!(retrieved.conversation_id, created.conversation_id);
}

#[tokio::test]
async fn test_conversation_store_persist_and_rehydrate() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let conversation = store.create().await.expect("create failed");
    let conv_id = &conversation.conversation_id;

    let items = vec![create_input_item("hello"), create_output_item("msg_1")];

    let metadata = ResponseMetadata::default();

    store
        .persist(conv_id, "resp_1", None, items, &metadata)
        .await
        .expect("persist failed");

    let rehydrated = store.rehydrate(conv_id).await.expect("rehydrate failed");

    assert_eq!(rehydrated.len(), 2);
}

#[tokio::test]
async fn conversation_snapshot_reports_empty_and_last_response() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);
    let conversation = store.create().await?;

    let snapshot = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert!(snapshot.items.is_empty());
    assert_eq!(snapshot.version, ConversationVersion::Empty);

    store
        .persist(
            &conversation.conversation_id,
            "resp_1",
            None,
            vec![create_input_item("hello"), create_output_item("msg_1")],
            &ResponseMetadata::default(),
        )
        .await?;

    let snapshot = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert_eq!(snapshot.items.len(), 2);
    assert_eq!(
        snapshot.version,
        ConversationVersion::LastResponse {
            response_id: "resp_1".to_owned(),
            last_sequence: Some(1),
        }
    );
    assert_eq!(store.rehydrate(&conversation.conversation_id).await?, snapshot.items);

    Ok(())
}

#[tokio::test]
async fn conversation_snapshot_version_includes_an_undecodable_final_row() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await?;
    let stored_item = create_input_item("hello");

    store
        .persist(
            &conversation.conversation_id,
            "resp_1",
            None,
            vec![stored_item.clone()],
            &ResponseMetadata::default(),
        )
        .await?;
    sqlx::query("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES ($1, $2, $3, $4, $5)")
        .bind("item_undecodable")
        .bind("not valid JSON")
        .bind(0_i64)
        .bind(&conversation.conversation_id)
        .bind(1_i64)
        .execute(pool.as_ref())
        .await?;

    let snapshot = store.rehydrate_snapshot(&conversation.conversation_id).await?;

    assert_eq!(snapshot.items, vec![stored_item]);
    assert_eq!(
        snapshot.version,
        ConversationVersion::LastResponse {
            response_id: "resp_1".to_owned(),
            last_sequence: Some(1),
        }
    );

    Ok(())
}

#[tokio::test]
async fn conversation_snapshot_rejects_items_without_a_sequence() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await?;

    store
        .persist(
            &conversation.conversation_id,
            "resp_1",
            None,
            vec![create_input_item("hello")],
            &ResponseMetadata::default(),
        )
        .await?;

    let item_id: String = sqlx::query_scalar("SELECT id FROM items WHERE conversation_id = $1")
        .bind(&conversation.conversation_id)
        .fetch_one(pool.as_ref())
        .await?;
    sqlx::query("UPDATE items SET seq = NULL WHERE id = $1")
        .bind(&item_id)
        .execute(pool.as_ref())
        .await?;

    let error = store
        .rehydrate_snapshot(&conversation.conversation_id)
        .await
        .expect_err("snapshot must reject an item without a sequence");
    assert!(matches!(
        error,
        StorageError::InvalidConversationSequence {
            conversation_id,
            item_id: invalid_item_id,
        } if conversation_id == conversation.conversation_id && invalid_item_id == item_id
    ));

    Ok(())
}

#[tokio::test]
async fn legacy_item_only_version_upgrades_on_the_next_persist() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await?;
    let legacy_item = create_input_item("legacy");
    sqlx::query("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES ($1, $2, $3, $4, $5)")
        .bind("item_legacy")
        .bind(String::try_from(&legacy_item)?)
        .bind(0_i64)
        .bind(&conversation.conversation_id)
        .bind(0_i64)
        .execute(pool.as_ref())
        .await?;

    let legacy = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert_eq!(legacy.version, ConversationVersion::LastSequence(0));

    store
        .persist_if_version(
            &conversation.conversation_id,
            legacy.version,
            "resp_after_legacy",
            None,
            Vec::new(),
            &ResponseMetadata::default(),
        )
        .await?;
    let upgraded = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert_eq!(
        upgraded.version,
        ConversationVersion::LastResponse {
            response_id: "resp_after_legacy".to_owned(),
            last_sequence: Some(0),
        }
    );

    Ok(())
}

#[tokio::test]
async fn conversation_version_empty_checked_persist_succeeds() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);
    let conversation = store.create().await?;
    let items = vec![create_input_item("first input"), create_output_item("msg_first")];

    store
        .persist_if_version(
            &conversation.conversation_id,
            ConversationVersion::Empty,
            "resp_first",
            None,
            items.clone(),
            &ResponseMetadata::default(),
        )
        .await?;

    let snapshot = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert_eq!(snapshot.items, items);
    assert_eq!(
        snapshot.version,
        ConversationVersion::LastResponse {
            response_id: "resp_first".to_owned(),
            last_sequence: Some(1),
        }
    );

    Ok(())
}

#[tokio::test]
async fn zero_item_turn_advances_version_and_retains_exact_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let response_store = ResponseStore::new(pool);
    let conversation = store.create().await?;
    let empty = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    let first_metadata = ResponseMetadata {
        model: "first-zero-item-turn".to_owned(),
        ..ResponseMetadata::default()
    };

    store
        .persist_if_version(
            &conversation.conversation_id,
            empty.version,
            "resp_zero_items_first",
            None,
            Vec::new(),
            &first_metadata,
        )
        .await?;

    let first = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert!(first.items.is_empty());
    assert_eq!(
        first.version,
        ConversationVersion::LastResponse {
            response_id: "resp_zero_items_first".to_owned(),
            last_sequence: None,
        }
    );
    assert_eq!(
        store
            .response_metadata_at_version(&conversation.conversation_id, &first.version)
            .await?
            .map(|metadata| metadata.model)
            .as_deref(),
        Some("first-zero-item-turn")
    );

    store
        .persist_if_version(
            &conversation.conversation_id,
            first.version.clone(),
            "resp_zero_items_second",
            None,
            Vec::new(),
            &ResponseMetadata {
                model: "second-zero-item-turn".to_owned(),
                ..ResponseMetadata::default()
            },
        )
        .await?;

    let second = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    assert!(second.items.is_empty());
    assert_ne!(second.version, first.version);
    assert_eq!(
        store
            .response_metadata_at_version(&conversation.conversation_id, &first.version)
            .await?
            .map(|metadata| metadata.model)
            .as_deref(),
        Some("first-zero-item-turn")
    );
    assert_eq!(
        store
            .response_metadata_at_version(&conversation.conversation_id, &second.version)
            .await?
            .map(|metadata| metadata.model)
            .as_deref(),
        Some("second-zero-item-turn")
    );

    let error = store
        .persist_if_version(
            &conversation.conversation_id,
            first.version,
            "resp_zero_items_stale",
            None,
            Vec::new(),
            &ResponseMetadata::default(),
        )
        .await
        .expect_err("an earlier zero-item turn must be stale");
    assert!(error.is_conversation_conflict());
    assert!(response_store.get("resp_zero_items_stale").await.is_err());

    Ok(())
}

#[tokio::test]
async fn conversation_version_stale_checked_persist_rolls_back_items_and_response()
-> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let response_store = ResponseStore::new(Arc::clone(&pool));
    let conversation = store.create().await?;
    let snapshot = store.rehydrate_snapshot(&conversation.conversation_id).await?;
    let competing_items = vec![
        create_input_item("competing input"),
        create_output_item("msg_competing"),
    ];
    store
        .persist(
            &conversation.conversation_id,
            "resp_competing",
            None,
            competing_items.clone(),
            &ResponseMetadata::default(),
        )
        .await?;
    let rejected_items = vec![create_input_item("stale input"), create_output_item("msg_stale")];

    let error = store
        .persist_if_version(
            &conversation.conversation_id,
            snapshot.version,
            "resp_stale",
            None,
            rejected_items,
            &ResponseMetadata::default(),
        )
        .await
        .expect_err("a stale conversation version must be rejected");

    assert!(error.is_conversation_conflict());
    assert!(matches!(
        error,
        StorageError::ConversationConflict { conversation_id }
            if conversation_id == conversation.conversation_id
    ));
    assert_eq!(store.rehydrate(&conversation.conversation_id).await?, competing_items);
    let response_error = response_store
        .get("resp_stale")
        .await
        .expect_err("the rejected response must not be stored");
    assert!(response_error.is_not_found());

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn conversation_version_racing_checked_persists_allow_exactly_one_winner()
-> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await?;
    let version = store.rehydrate_snapshot(&conversation.conversation_id).await?.version;
    let barrier = Arc::new(tokio::sync::Barrier::new(2));

    let writer_one = {
        let store = ConversationStore::new(Arc::clone(&pool));
        let conversation_id = conversation.conversation_id.clone();
        let barrier = Arc::clone(&barrier);
        let version = version.clone();
        tokio::spawn(async move {
            let items = vec![create_input_item("writer one"), create_output_item("msg_writer_one")];
            barrier.wait().await;
            let result = store
                .persist_if_version(
                    &conversation_id,
                    version,
                    "resp_writer_one",
                    None,
                    items.clone(),
                    &ResponseMetadata::default(),
                )
                .await;
            (result, items)
        })
    };
    let writer_two = {
        let store = ConversationStore::new(pool);
        let conversation_id = conversation.conversation_id.clone();
        tokio::spawn(async move {
            let items = vec![create_input_item("writer two"), create_output_item("msg_writer_two")];
            barrier.wait().await;
            let result = store
                .persist_if_version(
                    &conversation_id,
                    version,
                    "resp_writer_two",
                    None,
                    items.clone(),
                    &ResponseMetadata::default(),
                )
                .await;
            (result, items)
        })
    };

    let (writer_one_result, writer_one_items) = writer_one.await?;
    let (writer_two_result, writer_two_items) = writer_two.await?;

    assert_eq!(
        usize::from(writer_one_result.is_ok()) + usize::from(writer_two_result.is_ok()),
        1
    );
    assert_eq!(
        usize::from(
            writer_one_result
                .as_ref()
                .is_err_and(StorageError::is_conversation_conflict)
        ) + usize::from(
            writer_two_result
                .as_ref()
                .is_err_and(StorageError::is_conversation_conflict)
        ),
        1
    );
    let winner_items = if writer_one_result.is_ok() {
        writer_one_items
    } else {
        writer_two_items
    };
    assert_eq!(store.rehydrate(&conversation.conversation_id).await?, winner_items);

    Ok(())
}

#[tokio::test]
async fn conversation_version_is_scoped_per_conversation() -> Result<(), Box<dyn std::error::Error>> {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);
    let first = store.create().await?;
    let second = store.create().await?;
    let first_items = vec![create_input_item("first conversation")];
    store
        .persist(
            &first.conversation_id,
            "resp_first_conversation",
            None,
            first_items.clone(),
            &ResponseMetadata::default(),
        )
        .await?;
    let first_snapshot = store.rehydrate_snapshot(&first.conversation_id).await?;

    store
        .persist_if_version(
            &second.conversation_id,
            ConversationVersion::Empty,
            "resp_second_conversation",
            None,
            vec![create_input_item("second conversation")],
            &ResponseMetadata::default(),
        )
        .await?;

    let first_after = store.rehydrate_snapshot(&first.conversation_id).await?;
    assert_eq!(first_after.items, first_items);
    assert_eq!(first_after.version, first_snapshot.version);

    Ok(())
}

#[tokio::test]
async fn test_conversation_store_multiple_turns() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let conversation = store.create().await.expect("create failed");
    let conv_id = &conversation.conversation_id;

    let metadata = ResponseMetadata::default();

    // First turn
    store
        .persist(conv_id, "resp_1", None, vec![create_input_item("turn 1")], &metadata)
        .await
        .expect("first persist failed");

    // Second turn
    store
        .persist(
            conv_id,
            "resp_2",
            Some("resp_1"),
            vec![create_input_item("turn 2")],
            &metadata,
        )
        .await
        .expect("second persist failed");

    let rehydrated = store.rehydrate(conv_id).await.expect("rehydrate failed");

    assert_eq!(rehydrated.len(), 2);
}

#[tokio::test]
async fn test_response_store_persist_and_rehydrate() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let items = vec![create_input_item("query"), create_output_item("out_1")];

    let metadata = ResponseMetadata::default();

    store
        .persist("resp_1", None, items, &metadata)
        .await
        .expect("persist failed");

    let rehydrated = store.rehydrate("resp_1").await.expect("rehydrate failed");

    assert_eq!(rehydrated.len(), 2);
}

#[tokio::test]
async fn test_response_store_get() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let items = vec![create_input_item("test")];
    let metadata = ResponseMetadata::default();

    store
        .persist("resp_get_test", None, items, &metadata)
        .await
        .expect("persist failed");

    let response = store.get("resp_get_test").await.expect("get failed");

    assert_eq!(response.response_id, "resp_get_test");
    assert_eq!(response.history_item_ids.len(), 1);
}

#[tokio::test]
async fn test_response_store_with_previous_response() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let metadata = ResponseMetadata::default();

    store
        .persist("resp_1", None, vec![create_input_item("first")], &metadata)
        .await
        .expect("persist first failed");

    store
        .persist("resp_2", Some("resp_1"), vec![create_output_item("out_2")], &metadata)
        .await
        .expect("persist second failed");

    let response = store.get("resp_2").await.expect("get failed");

    assert_eq!(response.previous_response_id, Some("resp_1".to_string()));
    assert_eq!(response.history_item_ids.len(), 2);

    let rehydrated = store.rehydrate("resp_2").await.expect("rehydrate failed");
    assert_eq!(rehydrated.len(), 2);
}

// Edge case tests

#[tokio::test]
async fn test_conversation_persist_empty_items() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let conversation = store.create().await.expect("create failed");
    let conv_id = &conversation.conversation_id;

    let metadata = ResponseMetadata::default();

    // Persist with empty item list
    store
        .persist(conv_id, "resp_empty", None, vec![], &metadata)
        .await
        .expect("persist empty items failed");

    let rehydrated = store.rehydrate(conv_id).await.expect("rehydrate failed");

    assert!(rehydrated.is_empty());
}

#[tokio::test]
async fn test_conversation_rehydrate_after_multiple_varying_turns() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let conversation = store.create().await.expect("create failed");
    let conv_id = &conversation.conversation_id;

    let metadata = ResponseMetadata::default();

    // Turn 1: 1 item
    store
        .persist(conv_id, "resp_1", None, vec![create_input_item("turn1")], &metadata)
        .await
        .expect("turn 1 failed");

    // Turn 2: 3 items
    store
        .persist(
            conv_id,
            "resp_2",
            Some("resp_1"),
            vec![
                create_input_item("turn2a"),
                create_output_item("out2"),
                create_input_item("turn2b"),
            ],
            &metadata,
        )
        .await
        .expect("turn 2 failed");

    // Turn 3: 2 items
    store
        .persist(
            conv_id,
            "resp_3",
            Some("resp_2"),
            vec![create_input_item("turn3"), create_output_item("out3")],
            &metadata,
        )
        .await
        .expect("turn 3 failed");

    let rehydrated = store.rehydrate(conv_id).await.expect("rehydrate failed");

    assert_eq!(rehydrated.len(), 6);
}

#[tokio::test]
async fn test_response_store_chaining_respects_foreign_key() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let metadata = ResponseMetadata::default();

    // Create resp_1
    store
        .persist("resp_1", None, vec![create_input_item("first")], &metadata)
        .await
        .expect("resp_1 persist failed");

    // Try to create resp_3 with resp_2 as previous (resp_2 doesn't exist)
    // This should fail due to foreign key constraint
    let result = store
        .persist("resp_3", Some("resp_2"), vec![create_output_item("out3")], &metadata)
        .await;

    assert!(
        result.is_err(),
        "expected error when previous_response_id references non-existent response"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_conversation_concurrent_turns() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool.clone());

    let conversation = store.create().await.expect("create failed");
    let conv_id = conversation.conversation_id.clone();

    let metadata_1 = Arc::new(ResponseMetadata::default());
    let metadata_2 = metadata_1.clone();

    // Spawn two concurrent persist operations
    let conv_id_1 = conv_id.clone();
    let store_1 = ConversationStore::new(pool.clone());
    let handle1 = tokio::spawn(async move {
        store_1
            .persist(
                &conv_id_1,
                "resp_t1",
                None,
                vec![create_input_item("thread1")],
                metadata_1.as_ref(),
            )
            .await
    });

    let conv_id_2 = conv_id.clone();
    let store_2 = ConversationStore::new(pool);
    let handle2 = tokio::spawn(async move {
        store_2
            .persist(
                &conv_id_2,
                "resp_t2",
                None,
                vec![create_input_item("thread2")],
                metadata_2.as_ref(),
            )
            .await
    });

    let result1 = handle1.await;
    let result2 = handle2.await;

    assert!(result1.is_ok() && result1.unwrap().is_ok());
    assert!(result2.is_ok() && result2.unwrap().is_ok());

    let rehydrated = store.rehydrate(&conv_id).await.expect("rehydrate failed");
    assert_eq!(rehydrated.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_sqlite_multi_pool_mixed_read_write_concurrency() {
    let db_path = std::env::temp_dir().join(format!("mixed_rw_{}.db", uuid::Uuid::now_v7()));
    let db_url = format!("sqlite://{}", db_path.display());

    let writer_pool_a = create_pool_with_schema(Some(&db_url))
        .await
        .expect("failed to create writer pool a");
    let writer_pool_b = create_pool_with_schema(Some(&db_url))
        .await
        .expect("failed to create writer pool b");
    let reader_pool = create_pool_with_schema(Some(&db_url))
        .await
        .expect("failed to create reader pool");

    let writer_store_a = ConversationStore::new(Arc::clone(&writer_pool_a));
    let writer_store_b = ConversationStore::new(writer_pool_b);
    let reader_store = ConversationStore::new(reader_pool);
    let conversation = writer_store_a.create().await.expect("create conversation failed");
    let conv_id = conversation.conversation_id;
    let metadata = Arc::new(ResponseMetadata::default());
    let barrier = Arc::new(tokio::sync::Barrier::new(10));

    let spawn_writer = |writer_idx: usize, writer_store: ConversationStore| {
        let writer_conv_id = conv_id.clone();
        let writer_metadata = Arc::clone(&metadata);
        let writer_barrier = Arc::clone(&barrier);
        tokio::spawn(async move {
            writer_barrier.wait().await;
            for idx in 0..50 {
                writer_store
                    .persist(
                        &writer_conv_id,
                        &format!("resp_lock_writer_{writer_idx}_{idx}"),
                        None,
                        vec![create_input_item(&format!("writer {writer_idx} item {idx}"))],
                        writer_metadata.as_ref(),
                    )
                    .await
                    .map_err(|err| format!("writer {writer_idx} write {idx} failed: {err:?}"))?;
                tokio::task::yield_now().await;
            }
            Ok::<(), String>(())
        })
    };
    let writers = vec![spawn_writer(0, writer_store_a.clone()), spawn_writer(1, writer_store_b)];

    let mut readers = Vec::new();
    for reader_idx in 0..8 {
        let reader_store = reader_store.clone();
        let reader_conv_id = conv_id.clone();
        let reader_barrier = Arc::clone(&barrier);
        readers.push(tokio::spawn(async move {
            reader_barrier.wait().await;
            for iter in 0..100 {
                reader_store
                    .rehydrate(&reader_conv_id)
                    .await
                    .map_err(|err| format!("reader {reader_idx} iteration {iter} failed: {err:?}"))?;
                tokio::task::yield_now().await;
            }
            Ok::<(), String>(())
        }));
    }

    for writer in writers {
        writer.await.expect("writer task panicked").expect("writer task failed");
    }
    for reader in readers {
        reader.await.expect("reader task panicked").expect("reader task failed");
    }

    let final_items = ConversationStore::new(Arc::clone(&writer_pool_a))
        .rehydrate(&conv_id)
        .await
        .expect("final rehydrate failed");
    assert_eq!(final_items.len(), 100);

    let seqs: Vec<i64> = sqlx::query_scalar("SELECT seq FROM items WHERE conversation_id = ? ORDER BY seq ASC")
        .bind(&conv_id)
        .fetch_all(writer_pool_a.as_ref())
        .await
        .expect("sequence query failed");
    assert_eq!(seqs, (0..100).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_sqlite_same_pool_mixed_read_write_concurrency() {
    let db_path = std::env::temp_dir().join(format!("same_pool_mixed_rw_{}.db", uuid::Uuid::now_v7()));
    let db_url = format!("sqlite://{}", db_path.display());
    let sqlite_config = SqliteConfig {
        max_connections: 4,
        ..SqliteConfig::default()
    };
    let pool = create_pool_with_schema_and_sqlite_config(Some(&db_url), sqlite_config)
        .await
        .expect("failed to create pool");
    assert_eq!(pool.options().get_max_connections(), 4);

    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await.expect("create conversation failed");
    let conv_id = conversation.conversation_id;
    let metadata = Arc::new(ResponseMetadata::default());
    let barrier = Arc::new(tokio::sync::Barrier::new(10));

    let spawn_writer = |writer_idx: usize| {
        let writer_store = store.clone();
        let writer_conv_id = conv_id.clone();
        let writer_metadata = Arc::clone(&metadata);
        let writer_barrier = Arc::clone(&barrier);
        tokio::spawn(async move {
            writer_barrier.wait().await;
            for idx in 0..50 {
                writer_store
                    .persist(
                        &writer_conv_id,
                        &format!("resp_same_pool_writer_{writer_idx}_{idx}"),
                        None,
                        vec![create_input_item(&format!("same pool writer {writer_idx} item {idx}"))],
                        writer_metadata.as_ref(),
                    )
                    .await
                    .map_err(|err| format!("writer {writer_idx} write {idx} failed: {err:?}"))?;
                tokio::task::yield_now().await;
            }
            Ok::<(), String>(())
        })
    };
    let writers = vec![spawn_writer(0), spawn_writer(1)];

    let mut readers = Vec::new();
    for reader_idx in 0..8 {
        let reader_store = store.clone();
        let reader_conv_id = conv_id.clone();
        let reader_barrier = Arc::clone(&barrier);
        readers.push(tokio::spawn(async move {
            reader_barrier.wait().await;
            for iter in 0..100 {
                reader_store
                    .rehydrate(&reader_conv_id)
                    .await
                    .map_err(|err| format!("reader {reader_idx} iteration {iter} failed: {err:?}"))?;
                tokio::task::yield_now().await;
            }
            Ok::<(), String>(())
        }));
    }

    for writer in writers {
        writer.await.expect("writer task panicked").expect("writer task failed");
    }
    for reader in readers {
        reader.await.expect("reader task panicked").expect("reader task failed");
    }

    let final_items = store.rehydrate(&conv_id).await.expect("final rehydrate failed");
    assert_eq!(final_items.len(), 100);

    let seqs: Vec<i64> = sqlx::query_scalar("SELECT seq FROM items WHERE conversation_id = ? ORDER BY seq ASC")
        .bind(&conv_id)
        .fetch_all(pool.as_ref())
        .await
        .expect("sequence query failed");
    assert_eq!(seqs, (0..100).collect::<Vec<_>>());
}

// Store-level error handling edge cases

#[tokio::test]
async fn test_conversation_store_get_nonexistent() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let result = store.get("nonexistent_conv").await;
    assert!(result.is_err(), "expected error for non-existent conversation");

    // Verify it's a not found error
    let err = result.unwrap_err();
    assert!(err.is_not_found());
}

#[tokio::test]
async fn test_conversation_store_persist_nonexistent_conversation() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let metadata = ResponseMetadata::default();

    // Try to persist to a non-existent conversation
    let result = store
        .persist(
            "nonexistent_conv",
            "resp_1",
            None,
            vec![create_input_item("test")],
            &metadata,
        )
        .await;

    let error = result.expect_err("persisting to a non-existent conversation should fail");
    assert!(error.is_not_found(), "expected not-found error, got {error}");
}

#[tokio::test]
async fn test_response_store_rehydrate_nonexistent() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let result = store.rehydrate("nonexistent_resp").await;
    assert!(result.is_err(), "expected error for non-existent response");
}

#[tokio::test]
async fn test_conversation_store_disabled() {
    let store = ConversationStore::disabled();

    let result = store.create().await;
    assert!(result.is_err(), "expected error from disabled store");

    let err = result.unwrap_err();
    assert!(err.is_not_configured());
}

#[tokio::test]
async fn test_response_store_disabled() {
    let store = ResponseStore::disabled();

    let metadata = ResponseMetadata::default();
    let result = store
        .persist("resp_1", None, vec![create_input_item("test")], &metadata)
        .await;

    assert!(result.is_err(), "expected error from disabled store");

    let err = result.unwrap_err();
    assert!(err.is_not_configured());
}

#[tokio::test]
async fn test_conversation_store_get_after_create() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let created = store.create().await.expect("create failed");

    // Immediately try to get it
    let retrieved = store.get(&created.conversation_id).await.expect("get should succeed");

    assert_eq!(retrieved.conversation_id, created.conversation_id);
    assert_eq!(retrieved.created_at, created.created_at);
}

#[tokio::test]
async fn test_response_store_get_after_persist() {
    let pool = setup_pool().await;
    let store = ResponseStore::new(pool);

    let items = vec![create_input_item("query"), create_output_item("out_1")];
    let metadata = ResponseMetadata::default();

    store
        .persist("resp_stored", None, items.clone(), &metadata)
        .await
        .expect("persist failed");

    let retrieved = store.get("resp_stored").await.expect("response should be found");

    assert_eq!(retrieved.response_id, "resp_stored");
    assert_eq!(retrieved.history_item_ids.len(), 2);
}

#[tokio::test]
async fn test_tool_search_conversation_metadata_matches_snapshot_version() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(Arc::clone(&pool));
    let conversation = store.create().await.expect("create conversation");
    store
        .persist(
            &conversation.conversation_id,
            "resp_tool_search_initial",
            None,
            vec![create_input_item("initial")],
            &ResponseMetadata {
                model: "initial-model".to_owned(),
                ..ResponseMetadata::default()
            },
        )
        .await
        .expect("persist initial turn");

    let loaded_tool: agentic_core::types::tools::ResponsesTool = serde_json::from_value(serde_json::json!({
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object"},
        "defer_loading": true
    }))
    .expect("valid loaded function");
    let latest = ResponseMetadata {
        model: "latest-model".to_owned(),
        effective_tools: Some(vec![loaded_tool.clone()]),
        tool_search_loaded_tools: Some(vec![loaded_tool]),
        ..ResponseMetadata::default()
    };
    store
        .persist(
            &conversation.conversation_id,
            "resp_tool_search_latest",
            None,
            vec![create_input_item("latest")],
            &latest,
        )
        .await
        .expect("persist latest turn");

    // Simulate replicas whose clocks and process-local UUID order disagree with committed conversation-item sequence.
    sqlx::query("UPDATE responses SET created_at = $1 WHERE id = $2")
        .bind(9_999_i64)
        .bind("resp_tool_search_initial")
        .execute(pool.as_ref())
        .await
        .expect("make older turn appear newer by wall clock");
    sqlx::query("UPDATE responses SET created_at = $1 WHERE id = $2")
        .bind(1_i64)
        .bind("resp_tool_search_latest")
        .execute(pool.as_ref())
        .await
        .expect("make latest turn appear older by wall clock");

    let latest_item_id: String =
        sqlx::query_scalar("SELECT id FROM items WHERE conversation_id = $1 ORDER BY seq DESC LIMIT 1")
            .bind(&conversation.conversation_id)
            .fetch_one(pool.as_ref())
            .await
            .expect("latest conversation item");
    let branch = ResponseMetadata {
        model: "branch-model".to_owned(),
        ..ResponseMetadata::default()
    };
    sqlx::query(
        "INSERT INTO responses \
         (id, conversation_id, previous_response_id, history_item_ids, metadata, created_at) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind("resp_tool_search_branch")
    .bind(&conversation.conversation_id)
    .bind("resp_tool_search_latest")
    .bind(serde_json::to_string(&vec![latest_item_id]).expect("branch history JSON"))
    .bind(String::try_from(&branch).expect("branch metadata JSON"))
    .bind(20_000_i64)
    .execute(pool.as_ref())
    .await
    .expect("seed conversation-tagged response branch");

    let snapshot = store
        .rehydrate_snapshot(&conversation.conversation_id)
        .await
        .expect("rehydrate typed snapshot");
    store
        .persist(
            &conversation.conversation_id,
            "resp_after_snapshot",
            None,
            vec![create_input_item("after snapshot")],
            &ResponseMetadata {
                model: "after-snapshot-model".to_owned(),
                ..ResponseMetadata::default()
            },
        )
        .await
        .expect("persist a newer conversation turn");
    let metadata = store
        .response_metadata_at_version(&conversation.conversation_id, &snapshot.version)
        .await
        .expect("load response metadata at snapshot version")
        .expect("response metadata accompanies conversation items");
    assert_eq!(metadata.model, "latest-model");
    assert_eq!(metadata.tool_search_loaded_tools.as_deref().map(<[_]>::len), Some(1));
}

#[tokio::test]
async fn test_tool_search_conversation_conflict_does_not_persist_stale_loaded_state() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);
    let conversation = store.create().await.expect("create conversation");
    let snapshot = store
        .rehydrate_snapshot(&conversation.conversation_id)
        .await
        .expect("capture empty version");
    let winning_tool: agentic_core::types::tools::ResponsesTool = serde_json::from_value(serde_json::json!({
        "type": "function",
        "name": "winning_tool",
        "parameters": {"type": "object"},
        "defer_loading": true
    }))
    .expect("winning tool");
    let stale_tool: agentic_core::types::tools::ResponsesTool = serde_json::from_value(serde_json::json!({
        "type": "function",
        "name": "stale_tool",
        "parameters": {"type": "object"},
        "defer_loading": true
    }))
    .expect("stale tool");
    let winning = ResponseMetadata {
        model: "winner".to_owned(),
        effective_tools: Some(vec![winning_tool.clone()]),
        tool_search_loaded_tools: Some(vec![winning_tool]),
        ..ResponseMetadata::default()
    };
    let stale = ResponseMetadata {
        model: "stale".to_owned(),
        effective_tools: Some(vec![stale_tool.clone()]),
        tool_search_loaded_tools: Some(vec![stale_tool]),
        ..ResponseMetadata::default()
    };

    store
        .persist_if_version(
            &conversation.conversation_id,
            snapshot.version.clone(),
            "resp_tool_search_winner",
            None,
            vec![create_input_item("winner")],
            &winning,
        )
        .await
        .expect("winning turn persists");
    let error = store
        .persist_if_version(
            &conversation.conversation_id,
            snapshot.version,
            "resp_tool_search_stale",
            None,
            vec![create_input_item("stale")],
            &stale,
        )
        .await
        .expect_err("stale turn conflicts");
    assert!(matches!(error, StorageError::ConversationConflict { .. }));

    let snapshot = store
        .rehydrate_snapshot(&conversation.conversation_id)
        .await
        .expect("rehydrate winning state");
    let latest = store
        .response_metadata_at_version(&conversation.conversation_id, &snapshot.version)
        .await
        .expect("load winning metadata")
        .expect("winning metadata");
    assert_eq!(latest.model, "winner");
    let serialized = serde_json::to_value(latest.tool_search_loaded_tools).unwrap();
    assert_eq!(serialized[0]["name"], "winning_tool");
    assert!(!serialized.to_string().contains("stale_tool"));
}

#[tokio::test]
async fn test_conversation_get_or_create_same_id() {
    let pool = setup_pool().await;
    let store = ConversationStore::new(pool);

    let conv_id = "test_conv_idempotent";

    let first = store.get_or_create(conv_id).await.expect("first get_or_create failed");

    let second = store.get_or_create(conv_id).await.expect("second get_or_create failed");

    // Should return the same conversation
    assert_eq!(first.conversation_id, second.conversation_id);
    assert_eq!(first.created_at, second.created_at);
}
