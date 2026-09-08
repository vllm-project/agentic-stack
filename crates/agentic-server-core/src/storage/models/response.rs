//! LLM API response stored in the database.

use super::super::pool::{DbPool, DbResult, DbTransaction};
use crate::utils::common::{deserialize_from_string_opt, deserialize_from_string_opt_or_default, utcnow_str};

/// LLM API response stored in the database.
///
/// Maps to the `responses` table and represents a single API response
/// with its metadata and history chain.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Response {
    /// Unique response identifier.
    pub id: String,

    /// Optional conversation this response belongs to.
    pub conversation_id: Option<String>,

    /// Optional reference to previous response for chaining.
    pub previous_response_id: Option<String>,

    /// History item IDs as JSON array string.
    pub history_item_ids: Option<String>,

    /// Response metadata as JSON object string.
    pub metadata: Option<String>,

    /// Creation timestamp as Unix timestamp in seconds.
    pub created_at: i64,
}

/// Create a response in a transaction and return it.
///
/// # Errors
/// Returns `DbResult::Err` if the database insertion fails.
pub async fn create_in_tx(
    tx: &mut DbTransaction<'_>,
    id: &str,
    conversation_id: Option<&str>,
    previous_response_id: Option<&str>,
    history_item_ids: Option<&str>,
    metadata: Option<&str>,
) -> DbResult<Response> {
    let now = utcnow_str();
    sqlx::query_as::<_, Response>(
        "INSERT INTO responses \
         (id, conversation_id, previous_response_id, history_item_ids, metadata, created_at) \
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
    )
    .bind(id)
    .bind(conversation_id)
    .bind(previous_response_id)
    .bind(history_item_ids)
    .bind(metadata)
    .bind(now)
    .fetch_one(&mut **tx)
    .await
}

/// Get a response by ID.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get(pool: &DbPool, id: &str) -> DbResult<Option<Response>> {
    sqlx::query_as::<_, Response>("SELECT * FROM responses WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await
}

/// Get the response that committed a conversation turn.
///
/// # Errors
/// Returns `DbResult::Err` if the database query fails.
pub async fn get_conversation_turn(
    pool: &DbPool,
    conversation_id: &str,
    response_id: &str,
) -> DbResult<Option<Response>> {
    sqlx::query_as::<_, Response>(
        "SELECT * FROM responses \
         WHERE id = $1 AND conversation_id = $2",
    )
    .bind(response_id)
    .bind(conversation_id)
    .fetch_optional(pool)
    .await
}

impl Response {
    /// Deserialize `history_item_ids` from JSON string to Vec<String>.
    #[must_use]
    pub fn history_item_ids_vec(&self) -> Vec<String> {
        deserialize_from_string_opt_or_default(&self.history_item_ids)
    }

    /// Deserialize metadata from JSON string to the given type.
    #[must_use]
    pub fn metadata_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        deserialize_from_string_opt(&self.metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_history_ids_empty() {
        let response = Response {
            id: "test".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: None,
            created_at: 1_704_067_200,
        };

        let ids: Vec<String> = response.history_item_ids_vec();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_response_history_ids_valid() {
        let response = Response {
            id: "test".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: Some(r#"["item_1", "item_2"]"#.to_string()),
            metadata: None,
            created_at: 1_704_067_200,
        };

        let ids = response.history_item_ids_vec();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], "item_1");
    }

    #[test]
    fn test_response_metadata_deserialize() {
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct TestMeta {
            model: String,
        }

        let response = Response {
            id: "resp_1".to_string(),
            conversation_id: None,
            previous_response_id: None,
            history_item_ids: None,
            metadata: Some(r#"{"model":"gpt-4"}"#.to_string()),
            created_at: 1_704_067_200,
        };

        let meta: Option<TestMeta> = response.metadata_as();
        assert!(meta.is_some());
    }
}
