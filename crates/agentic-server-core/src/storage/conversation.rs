//! Conversation storage operations.

use std::convert::TryFrom;
use std::sync::Arc;

use super::models::{conversation, item, response};
use super::pool::DbPool;
use super::types::{
    ConversationData, ConversationSnapshot, ConversationVersion, InOutItem, ResponseMetadata, StorageError, StoreResult,
};
use crate::utils::common::{serialize_to_string, uuid7_str};

/// Conversation storage operations.
#[derive(Clone, Debug)]
pub struct ConversationStore {
    pool: Option<Arc<DbPool>>,
}

impl ConversationStore {
    /// Creates a disabled conversation store.
    #[must_use]
    pub fn disabled() -> Self {
        Self { pool: None }
    }

    /// Creates a new conversation store with database pool.
    #[must_use]
    pub fn new(pool: Arc<DbPool>) -> Self {
        Self { pool: Some(pool) }
    }

    /// Returns a reference to the database pool.
    ///
    /// # Errors
    ///
    /// Returns error if store is disabled (no pool configured).
    fn pool(&self) -> StoreResult<&DbPool> {
        self.pool.as_deref().ok_or(StorageError::NotConfigured)
    }

    /// Creates a new conversation.
    ///
    /// # Errors
    ///
    /// Returns error if database query fails.
    pub async fn create(&self) -> StoreResult<ConversationData> {
        let pool = self.pool()?;
        let row = conversation::create(pool, &uuid7_str("conv_")).await?;
        Ok(row.into())
    }

    /// Gets a conversation or creates it if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns error if database query fails.
    pub async fn get_or_create(&self, conversation_id: &str) -> StoreResult<ConversationData> {
        let pool = self.pool()?;
        let row = conversation::get_or_create(pool, conversation_id).await?;
        Ok(row.into())
    }

    /// Gets a conversation by ID.
    ///
    /// # Errors
    ///
    /// Returns error if conversation not found or database query fails.
    pub async fn get(&self, conversation_id: &str) -> StoreResult<ConversationData> {
        let pool = self.pool()?;
        let row = conversation::get(pool, conversation_id)
            .await?
            .ok_or_else(|| StorageError::not_found("Conversation", conversation_id))?;
        Ok(row.into())
    }

    /// Rehydrates a conversation with all its items.
    ///
    /// # Errors
    ///
    /// Returns an error if a stored item is missing its sequence number or if the database query fails.
    pub async fn rehydrate(&self, conversation_id: &str) -> StoreResult<Vec<InOutItem>> {
        Ok(self.rehydrate_snapshot(conversation_id).await?.items)
    }

    /// Rehydrates a conversation with its items and storage version.
    ///
    /// # Errors
    ///
    /// Returns an error if a stored item is missing its sequence number or if the database query fails.
    pub async fn rehydrate_snapshot(&self, conversation_id: &str) -> StoreResult<ConversationSnapshot> {
        let pool = self.pool()?;
        let snapshot_rows = conversation::get_snapshot(pool, conversation_id).await?;

        let mut last_sequence = None;
        for row in &snapshot_rows.items {
            last_sequence = Some(row.seq.ok_or_else(|| StorageError::InvalidConversationSequence {
                conversation_id: conversation_id.to_string(),
                item_id: row.id.clone(),
            })?);
        }

        Ok(ConversationSnapshot {
            items: snapshot_rows
                .items
                .into_iter()
                .filter_map(|row| row.as_inout())
                .collect(),
            version: ConversationVersion::from_snapshot(last_sequence, snapshot_rows.latest_response_id),
        })
    }

    /// Loads metadata from the persisted turn at a captured version.
    ///
    /// # Errors
    ///
    /// Returns an error if either targeted database lookup fails.
    pub async fn response_metadata_at_version(
        &self,
        conversation_id: &str,
        version: &ConversationVersion,
    ) -> StoreResult<Option<ResponseMetadata>> {
        let ConversationVersion::LastResponse { response_id, .. } = version else {
            return Ok(None);
        };
        let pool = self.pool()?;
        let response = response::get_conversation_turn(pool, conversation_id, response_id).await?;
        Ok(response.and_then(|row| row.metadata_as()))
    }

    /// Persists conversation turn with new items and response metadata.
    ///
    /// Creates items in the conversation and stores the associated response record.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if conversation not found or database operation fails.
    pub async fn persist(
        &self,
        conversation_id: &str,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: Vec<InOutItem>,
        metadata: &ResponseMetadata,
    ) -> StoreResult<()> {
        self.persist_impl(
            conversation_id,
            None,
            response_id,
            previous_response_id,
            new_items,
            metadata,
        )
        .await
    }

    /// Persists a conversation turn only if its stored version still matches.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] if the conversation changed, was not found, or a database operation fails.
    pub async fn persist_if_version(
        &self,
        conversation_id: &str,
        expected_version: ConversationVersion,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: Vec<InOutItem>,
        metadata: &ResponseMetadata,
    ) -> StoreResult<()> {
        self.persist_impl(
            conversation_id,
            Some(expected_version),
            response_id,
            previous_response_id,
            new_items,
            metadata,
        )
        .await
    }

    async fn persist_impl(
        &self,
        conversation_id: &str,
        expected_version: Option<ConversationVersion>,
        response_id: &str,
        previous_response_id: Option<&str>,
        new_items: Vec<InOutItem>,
        metadata: &ResponseMetadata,
    ) -> StoreResult<()> {
        let pool = self.pool()?;

        let mut item_ids: Vec<String> = Vec::new();
        let mut items_: Vec<(String, String)> = Vec::new();
        for any_item in new_items {
            let item_id = uuid7_str("item_");
            item_ids.push(item_id.clone());
            let data_str = String::try_from(&any_item)?;
            items_.push((item_id, data_str));
        }
        let history_item_ids_json = serialize_to_string(&item_ids)?;
        let metadata_json = String::try_from(metadata)?;

        let mut tx = pool.begin().await?;

        let locked_conversation = match conversation::lock_in_tx(&mut tx, conversation_id).await {
            Ok(conversation) => conversation,
            Err(sqlx::Error::RowNotFound) => {
                return Err(StorageError::not_found("Conversation", conversation_id));
            }
            Err(error) => return Err(error.into()),
        };
        if let Some(expected_version) = expected_version {
            let current_version = ConversationVersion::from_snapshot(
                item::last_conversation_sequence_in_tx(&mut tx, conversation_id).await?,
                locked_conversation.latest_response_id,
            );
            if current_version != expected_version {
                return Err(StorageError::ConversationConflict {
                    conversation_id: conversation_id.to_owned(),
                });
            }
        }
        item::create_in_tx(&mut tx, items_, Some(conversation_id)).await?;

        response::create_in_tx(
            &mut tx,
            response_id,
            Some(conversation_id),
            previous_response_id,
            Some(&history_item_ids_json),
            Some(&metadata_json),
        )
        .await?;
        conversation::set_latest_response_in_tx(&mut tx, conversation_id, response_id).await?;
        tx.commit().await?;

        Ok(())
    }
}
