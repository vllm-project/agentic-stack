//! Conversation history item stored in the database.

use serde_json::Value;
use std::fmt::Write;
use tracing::warn;

use super::super::pool::{DbPool, DbResult, DbTransaction};
use super::super::types::item::{InOutItem, ItemKind, STORED_ITEM_KIND_KEY};
use crate::types::io::{InputItem, OutputItem};
use crate::utils::common::{deserialize_from_str_opt, utcnow_str};

const ITEM_COLUMN_COUNT: usize = 5;
const SEQUENCE_COLUMN_INDEX: usize = 4;
const MAX_BIND_PARAMETERS: usize = 999;
const MAX_ITEMS_PER_INSERT: usize = MAX_BIND_PARAMETERS / ITEM_COLUMN_COUNT;

/// Conversation history item stored in the database.
///
/// Maps to the `items` table and represents a single message/event
/// in a conversation timeline.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Item {
    /// Unique identifier for this item.
    pub id: String,

    /// Item data stored as JSON text.
    /// Deserialized based on context (`message`, `tool_call`, etc.)
    pub data: String,

    /// Creation timestamp as Unix timestamp in seconds.
    pub created_at: i64,

    /// Optional conversation ID for grouping items.
    pub conversation_id: Option<String>,

    /// Optional sequence number within conversation.
    pub seq: Option<i64>,
}

impl Item {
    fn data_without_storage_marker(&self) -> Option<Value> {
        let mut value = deserialize_from_str_opt::<Value>(&self.data)?;
        if let Some(object) = value.as_object_mut() {
            object.remove(STORED_ITEM_KIND_KEY);
        }
        Some(value)
    }

    /// Deserialize data column as `InputItem`.
    #[must_use]
    pub fn as_input(&self) -> Option<InputItem> {
        serde_json::from_value(self.data_without_storage_marker()?).ok()
    }

    /// Deserialize data column as `OutputItem`.
    #[must_use]
    pub fn as_output(&self) -> Option<OutputItem> {
        serde_json::from_value(self.data_without_storage_marker()?).ok()
    }

    /// Deserialize data column as either `InputItem` or `OutputItem`.
    #[must_use]
    pub fn as_inout(&self) -> Option<InOutItem> {
        if let Some(kind) = self.stored_item_kind() {
            match kind {
                ItemKind::Input => {
                    if let Some(input) = self.as_input() {
                        return Some(InOutItem::Input(input));
                    }
                }
                ItemKind::Output => {
                    if let Some(output) = self.as_output() {
                        return Some(InOutItem::Output(output));
                    }
                }
            }
        }

        let output = self.as_output();
        if output.as_ref().is_some_and(|item| !matches!(item, OutputItem::Unknown)) {
            return output.map(InOutItem::Output);
        }

        let input = self.as_input();
        if input.as_ref().is_some_and(|item| !matches!(item, InputItem::Unknown)) {
            return input.map(InOutItem::Input);
        }

        match (input, output) {
            (Some(input), _) => Some(InOutItem::Input(input)),
            (_, Some(output)) => Some(InOutItem::Output(output)),
            _ => {
                warn!(item_id = %self.id, "unrecognized item type in stored data");
                None
            }
        }
    }

    fn stored_item_kind(&self) -> Option<ItemKind> {
        let value = deserialize_from_str_opt::<Value>(&self.data)?;
        ItemKind::from_stored_str(value.get(STORED_ITEM_KIND_KEY)?.as_str()?)
    }
}

fn item_values_clause(row_count: usize, first_bind_index: usize, sequence_from_cte: bool) -> String {
    let mut clause = String::new();
    let mut bind_index = first_bind_index;

    for row_index in 0..row_count {
        if row_index > 0 {
            clause.push_str(", ");
        }
        clause.push('(');
        for column_index in 0..ITEM_COLUMN_COUNT {
            if column_index > 0 {
                clause.push_str(", ");
            }
            if sequence_from_cte && column_index == SEQUENCE_COLUMN_INDEX {
                write!(clause, "(SELECT start + ${bind_index} FROM next_seq)").expect("writing to String cannot fail");
            } else {
                write!(clause, "${bind_index}").expect("writing to String cannot fail");
            }
            bind_index += 1;
        }
        clause.push(')');
    }

    clause
}

/// Create items in a transaction with optional conversation context.
///
/// If `conversation_id` is provided, the next sequence range is computed in the insert statement so
/// concurrent `SQLite` writers do not take a stale read snapshot before writing.
///
/// # Errors
/// Returns `DbResult::Err` if the database insertion fails.
pub async fn create_in_tx(
    tx: &mut DbTransaction<'_>,
    items: Vec<(String, String)>,
    conversation_id: Option<&str>,
) -> DbResult<Vec<Item>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut created = Vec::with_capacity(items.len());
    for batch in items.chunks(MAX_ITEMS_PER_INSERT) {
        let mut rows = if let Some(conversation_id) = conversation_id {
            create_in_tx_with_next_conversation_seq(tx, batch, conversation_id).await?
        } else {
            create_in_tx_without_conversation(tx, batch).await?
        };
        created.append(&mut rows);
    }
    Ok(created)
}

async fn create_in_tx_without_conversation(
    tx: &mut DbTransaction<'_>,
    items: &[(String, String)],
) -> DbResult<Vec<Item>> {
    let now = utcnow_str();
    let values_clause = item_values_clause(items.len(), 1, false);
    let sql =
        format!("INSERT INTO items (id, data, created_at, conversation_id, seq) VALUES {values_clause} RETURNING *");

    let mut query = sqlx::query_as::<_, Item>(&sql);
    for (id, data) in items {
        query = query.bind(id).bind(data).bind(now).bind(None::<&str>).bind(None::<i64>);
    }

    query.fetch_all(&mut **tx).await
}

async fn create_in_tx_with_next_conversation_seq(
    tx: &mut DbTransaction<'_>,
    items: &[(String, String)],
    conversation_id: &str,
) -> DbResult<Vec<Item>> {
    let now = utcnow_str();
    let values_clause = item_values_clause(items.len(), 2, true);
    let sql = format!(
        "WITH next_seq AS ( \
             SELECT COALESCE(MAX(seq), -1) + 1 AS start \
             FROM items \
             WHERE conversation_id = $1 \
         ) \
         INSERT INTO items (id, data, created_at, conversation_id, seq) \
         VALUES {values_clause} \
         RETURNING *"
    );

    let mut query = sqlx::query_as::<_, Item>(&sql).bind(conversation_id);
    #[allow(clippy::cast_possible_wrap)]
    for (idx, (id, data)) in items.iter().enumerate() {
        query = query
            .bind(id)
            .bind(data)
            .bind(now)
            .bind(conversation_id)
            .bind(idx as i64);
    }

    query.fetch_all(&mut **tx).await
}

/// Get items by IDs.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_items(pool: &DbPool, ids: &[String]) -> DbResult<Vec<Item>> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let mut rows = Vec::with_capacity(ids.len());
    for batch in ids.chunks(MAX_BIND_PARAMETERS) {
        let placeholders = (1..=batch.len())
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!("SELECT * FROM items WHERE id IN ({placeholders})");
        let mut query = sqlx::query_as::<_, Item>(&sql);
        for id in batch {
            query = query.bind(id);
        }
        rows.extend(query.fetch_all(pool).await?);
    }
    Ok(rows)
}

/// Get items by conversation ID ordered by sequence.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_items_by_conversation(pool: &DbPool, conversation_id: &str) -> DbResult<Vec<Item>> {
    sqlx::query_as::<_, Item>("SELECT * FROM items WHERE conversation_id = $1 ORDER BY seq ASC")
        .bind(conversation_id)
        .fetch_all(pool)
        .await
}

/// Returns the last stored item sequence for a conversation inside a transaction.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn last_conversation_sequence_in_tx(
    tx: &mut DbTransaction<'_>,
    conversation_id: &str,
) -> DbResult<Option<i64>> {
    sqlx::query_scalar("SELECT MAX(seq) FROM items WHERE conversation_id = $1")
        .bind(conversation_id)
        .fetch_one(&mut **tx)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::event::MessageStatus;
    use crate::types::io::{InputItem, OutputItem, ReasoningOutput, ReasoningTextContent};

    #[test]
    fn item_values_clause_numbers_plain_rows() {
        assert_eq!(
            item_values_clause(2, 1, false),
            "($1, $2, $3, $4, $5), ($6, $7, $8, $9, $10)"
        );
    }

    #[test]
    fn item_values_clause_numbers_conversation_rows_after_cte_bind() {
        assert_eq!(
            item_values_clause(2, 2, true),
            "($2, $3, $4, $5, (SELECT start + $6 FROM next_seq)), \
             ($7, $8, $9, $10, (SELECT start + $11 FROM next_seq))"
        );
    }

    #[tokio::test]
    async fn item_queries_chunk_above_portable_bind_limit() {
        let pool = crate::storage::create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create in-memory database");
        let items = (0..=MAX_BIND_PARAMETERS)
            .map(|index| (format!("item_{index}"), "{}".to_owned()))
            .collect::<Vec<_>>();
        let ids = items.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
        let mut transaction = pool.begin().await.expect("begin transaction");
        let created = create_in_tx(&mut transaction, items, None)
            .await
            .expect("insert item batches");
        transaction.commit().await.expect("commit item batches");
        let loaded = get_items(&pool, &ids).await.expect("load item batches");

        assert_eq!(created.len(), MAX_BIND_PARAMETERS + 1);
        assert_eq!(loaded.len(), MAX_BIND_PARAMETERS + 1);
    }

    #[tokio::test]
    async fn conversation_item_batches_keep_contiguous_sequences() {
        let pool = crate::storage::create_pool_with_schema(Some("sqlite://?mode=memory"))
            .await
            .expect("create in-memory database");
        let conversation_id = "conv_batch";
        crate::storage::models::conversation::create(&pool, conversation_id)
            .await
            .expect("create conversation");
        let item_count = MAX_ITEMS_PER_INSERT + 1;
        let items = (0..item_count)
            .map(|index| (format!("conversation_item_{index}"), "{}".to_owned()))
            .collect::<Vec<_>>();
        let mut transaction = pool.begin().await.expect("begin transaction");
        let created = create_in_tx(&mut transaction, items, Some(conversation_id))
            .await
            .expect("insert conversation item batches");
        transaction.commit().await.expect("commit item batches");
        let stored = get_items_by_conversation(&pool, conversation_id)
            .await
            .expect("load conversation item batches");
        let expected_sequences = (0..i64::try_from(item_count).expect("item count fits in i64")).collect::<Vec<_>>();

        assert_eq!(
            created
                .iter()
                .map(|item| item.seq.expect("created sequence"))
                .collect::<Vec<_>>(),
            expected_sequences
        );
        assert_eq!(
            stored
                .iter()
                .map(|item| item.seq.expect("stored sequence"))
                .collect::<Vec<_>>(),
            expected_sequences
        );
    }

    #[test]
    fn test_item_basic() {
        let item = Item {
            id: "item_123".to_string(),
            data: r#"{"role":"user","content":"hello"}"#.to_string(),
            created_at: 1_704_067_200,
            conversation_id: Some("conv_456".to_string()),
            seq: Some(1),
        };

        assert_eq!(item.id, "item_123");
        assert_eq!(item.conversation_id, Some("conv_456".to_string()));
        assert_eq!(item.seq, Some(1));
    }

    #[test]
    fn test_item_optional_fields() {
        let item = Item {
            id: "item_789".to_string(),
            data: r#"{"role":"assistant"}"#.to_string(),
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        assert!(item.conversation_id.is_none());
        assert!(item.seq.is_none());
    }

    #[test]
    fn complete_reasoning_round_trip_strips_storage_marker() {
        let mut reasoning = ReasoningOutput::new("rs_1");
        reasoning.content.extend([
            ReasoningTextContent::new("first thought"),
            ReasoningTextContent::new("second thought"),
        ]);
        reasoning
            .summary
            .push(serde_json::json!({"type": "summary_text", "text": "concise summary"}));
        reasoning.encrypted_content = Some(serde_json::json!({"ciphertext": "opaque"}));
        reasoning.status = Some("completed".to_owned());
        let stored = InOutItem::Output(OutputItem::Reasoning(reasoning));
        let stored_json = String::try_from(&stored).expect("serialization failed");
        assert!(stored_json.contains(STORED_ITEM_KIND_KEY));
        let item = Item {
            id: "item_reasoning".to_string(),
            data: stored_json,
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        let Some(InOutItem::Output(OutputItem::Reasoning(reasoning))) = item.as_inout() else {
            panic!("expected stored reasoning output");
        };
        assert_eq!(reasoning.id, "rs_1");
        assert_eq!(reasoning.content.len(), 2);
        assert_eq!(reasoning.summary[0]["text"], "concise summary");
        assert_eq!(
            reasoning.encrypted_content,
            Some(serde_json::json!({"ciphertext": "opaque"}))
        );
        assert_eq!(reasoning.status.as_deref(), Some("completed"));

        let reconstructed = serde_json::to_value(OutputItem::Reasoning(reasoning)).expect("reasoning value");
        assert!(reconstructed.get(STORED_ITEM_KIND_KEY).is_none());
    }

    #[test]
    fn test_legacy_output_message_rehydrates_as_output_before_unknown_input() {
        let item = Item {
            id: "item_message".to_string(),
            data: serde_json::json!({
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}]
            })
            .to_string(),
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        let stored = item.as_inout().expect("stored item");
        assert!(matches!(stored, InOutItem::Output(OutputItem::Message(_))));

        let inputs = InOutItem::into_input_items(vec![stored]);
        assert!(matches!(inputs[0], InputItem::Message(_)));
    }

    #[test]
    fn test_namespaced_function_call_rehydrates_without_storage_marker() {
        let stored = InOutItem::Output(OutputItem::FunctionCall(crate::types::io::FunctionToolCall {
            id: "fc_1".to_string(),
            call_id: "call_1".to_string(),
            name: "run".to_string(),
            namespace: Some("mcp__shell".to_string()),
            arguments: "{\"cmd\":\"pwd\"}".to_string(),
            status: MessageStatus::Completed,
        }));
        let item = Item {
            id: "item_function_call".to_string(),
            data: String::try_from(&stored).expect("serialization failed"),
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        let inputs = InOutItem::into_input_items(vec![item.as_inout().expect("stored item")]);
        let value = serde_json::to_value(&inputs[0]).expect("input value");

        assert_eq!(value["type"], "function_call");
        assert_eq!(value["namespace"], "mcp__shell");
        assert_eq!(value["name"], "run");
        assert!(value.get(STORED_ITEM_KIND_KEY).is_none());

        println!("namespace round-trip: mcp__shell.run -> storage -> input function_call");
        println!("storage marker stripped: _agentic_item_kind absent");
    }

    #[test]
    fn shell_call_round_trips_through_storage_and_rehydration() {
        let output: OutputItem = serde_json::from_value(serde_json::json!({
            "type": "shell_call",
            "id": "sh_1",
            "call_id": "call_shell",
            "action": {
                "commands": ["pwd"],
                "timeout_ms": 1_000,
                "max_output_length": 4_096
            },
            "status": "completed"
        }))
        .expect("shell output item");
        let stored = InOutItem::Output(output);
        let item = Item {
            id: "item_shell_call".to_owned(),
            data: String::try_from(&stored).expect("serialization failed"),
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        let inputs = InOutItem::into_input_items(vec![item.as_inout().expect("stored shell item")]);
        let value = serde_json::to_value(&inputs[0]).expect("rehydrated shell input");

        assert_eq!(value["type"], "shell_call");
        assert_eq!(value["call_id"], "call_shell");
        assert_eq!(value["action"]["commands"][0], "pwd");
        assert!(value.get(STORED_ITEM_KIND_KEY).is_none());
    }

    #[test]
    fn test_multiple_namespaced_function_calls_rehydrate_without_storage_marker() {
        let stored_items = [
            InOutItem::Output(OutputItem::FunctionCall(crate::types::io::FunctionToolCall {
                id: "fc_1".to_string(),
                call_id: "call_1".to_string(),
                name: "run".to_string(),
                namespace: Some("mcp__shell".to_string()),
                arguments: "{\"cmd\":\"pwd\"}".to_string(),
                status: MessageStatus::Completed,
            })),
            InOutItem::Output(OutputItem::FunctionCall(crate::types::io::FunctionToolCall {
                id: "fc_2".to_string(),
                call_id: "call_2".to_string(),
                name: "run".to_string(),
                namespace: Some("mcp__git".to_string()),
                arguments: "{\"args\":[\"status\",\"--short\"]}".to_string(),
                status: MessageStatus::Completed,
            })),
        ];
        let rows: Vec<InOutItem> = stored_items
            .iter()
            .enumerate()
            .map(|(idx, stored)| Item {
                id: format!("item_function_call_{idx}"),
                data: String::try_from(stored).expect("serialization failed"),
                created_at: 1_704_067_200,
                conversation_id: None,
                seq: Some(idx.try_into().expect("seq")),
            })
            .map(|item| item.as_inout().expect("stored item"))
            .collect();

        let inputs = InOutItem::into_input_items(rows);
        let values = serde_json::to_value(&inputs).expect("input values");

        assert_eq!(values[0]["type"], "function_call");
        assert_eq!(values[0]["namespace"], "mcp__shell");
        assert_eq!(values[0]["name"], "run");
        assert_eq!(values[0]["call_id"], "call_1");
        assert!(values[0].get(STORED_ITEM_KIND_KEY).is_none());

        assert_eq!(values[1]["type"], "function_call");
        assert_eq!(values[1]["namespace"], "mcp__git");
        assert_eq!(values[1]["name"], "run");
        assert_eq!(values[1]["call_id"], "call_2");
        assert!(values[1].get(STORED_ITEM_KIND_KEY).is_none());

        println!("namespace round-trip: mcp__shell.run -> call_1");
        println!("namespace round-trip: mcp__git.run -> call_2");
        println!("same tool name preserved under separate namespaces");
    }

    #[test]
    fn test_unknown_rehydrated_items_are_omitted() {
        let stored = InOutItem::Output(OutputItem::Unknown);
        let item = Item {
            id: "item_unknown".to_string(),
            data: String::try_from(&stored).expect("serialization failed"),
            created_at: 1_704_067_200,
            conversation_id: None,
            seq: None,
        };

        let inputs = InOutItem::into_input_items(vec![item.as_inout().expect("stored item")]);

        assert!(inputs.is_empty());
    }
}
