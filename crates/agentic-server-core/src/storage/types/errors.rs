//! Storage layer error types.

use serde_json;
use thiserror::Error;

/// Result type for storage operations.
///
/// All storage functions return `Result<T, StorageError>` for explicit error handling.
pub type StoreResult<T> = std::result::Result<T, StorageError>;

/// Storage layer errors with detailed context.
#[derive(Error, Debug)]
pub enum StorageError {
    /// Resource not found in database.
    #[error("not found: {resource_type} with id '{id}'")]
    NotFound { resource_type: String, id: String },

    /// A conversation item did not have its required sequence number.
    #[error("invalid conversation sequence for conversation '{conversation_id}' item '{item_id}'")]
    InvalidConversationSequence { conversation_id: String, item_id: String },

    /// A conversation changed after its version was read.
    #[error("conversation changed while the response was being generated")]
    ConversationConflict { conversation_id: String },

    /// Database operation failed.
    ///
    /// Wraps `sqlx::Error` and automatically converts from it via `#[from]`.
    /// This allows using `?` operator with sqlx results.
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Storage is not configured or disabled.
    #[error("storage not configured or disabled")]
    NotConfigured,

    /// Serialization or deserialization of data failed.
    ///
    /// Wraps `serde_json::Error` and automatically converts from it via `#[from]`.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl StorageError {
    /// Creates a "not found" error for a resource.
    #[must_use]
    pub fn not_found(resource_type: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }

    /// Returns `true` if this error is "not found".
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound { .. })
    }

    /// Returns `true` if this error is "not configured".
    #[must_use]
    pub fn is_not_configured(&self) -> bool {
        matches!(self, Self::NotConfigured)
    }

    /// Returns `true` if this error is a conversation version conflict.
    #[must_use]
    pub fn is_conversation_conflict(&self) -> bool {
        matches!(self, Self::ConversationConflict { .. })
    }

    /// Returns `true` when the database rejected a duplicate unique key.
    #[must_use]
    pub fn is_unique_violation(&self) -> bool {
        matches!(
            self,
            Self::Database(sqlx::Error::Database(error)) if error.is_unique_violation()
        )
    }

    /// Extracts the resource type and ID if this is a "not found" error.
    #[must_use]
    pub fn not_found_details(&self) -> Option<(String, String)> {
        match self {
            Self::NotFound { resource_type, id } => Some((resource_type.clone(), id.clone())),
            _ => None,
        }
    }

    /// Returns `true` if this error is a serialization error.
    #[must_use]
    pub fn is_serialization(&self) -> bool {
        matches!(self, Self::Serialization(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_found_error_creation() {
        let err = StorageError::not_found("Response", "resp_123");
        assert!(err.is_not_found());
        assert!(!err.is_not_configured());
    }

    #[test]
    fn test_not_found_details_extraction() {
        let err = StorageError::not_found("Agent", "agent_456");
        let details = err.not_found_details();
        assert!(details.is_some());
        let (resource, id) = details.unwrap();
        assert_eq!(resource, "Agent");
        assert_eq!(id, "agent_456");
    }

    #[test]
    fn test_not_configured_error() {
        let err = StorageError::NotConfigured;
        assert!(!err.is_not_found());
        assert!(err.is_not_configured());
    }

    #[test]
    fn test_error_display_formatting() {
        let err = StorageError::not_found("Response", "123");
        let msg = err.to_string();
        assert_eq!(msg, "not found: Response with id '123'");
    }
}
