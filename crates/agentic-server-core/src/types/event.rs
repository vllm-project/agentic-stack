//! Response and message status enums.

use std::convert::Infallible;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Response completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    /// Response is being generated.
    #[default]
    InProgress,

    /// Response generation completed successfully.
    Completed,

    /// Response generation incomplete (e.g., stream interrupted).
    Incomplete,

    /// Response generation encountered an error.
    Error,
}

impl ResponseStatus {
    /// Returns the canonical wire string for this status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Error => "error",
        }
    }
}

impl FromStr for ResponseStatus {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            _ => Self::Error,
        })
    }
}

/// Message item completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    /// Message is being generated.
    #[default]
    InProgress,

    /// Message generation completed.
    Completed,
}

impl MessageStatus {
    /// Returns the canonical wire string for this status.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
        }
    }
}

impl FromStr for MessageStatus {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "completed" => Self::Completed,
            _ => Self::InProgress,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_status_round_trip() {
        for (s, expected) in [
            ("in_progress", ResponseStatus::InProgress),
            ("completed", ResponseStatus::Completed),
            ("incomplete", ResponseStatus::Incomplete),
            ("error", ResponseStatus::Error),
        ] {
            let parsed: ResponseStatus = s.parse().unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(parsed.as_str(), s);
        }
    }

    #[test]
    fn test_message_status_round_trip() {
        assert_eq!("completed".parse::<MessageStatus>().unwrap(), MessageStatus::Completed);
        assert_eq!(
            "in_progress".parse::<MessageStatus>().unwrap(),
            MessageStatus::InProgress
        );
        assert_eq!("unknown".parse::<MessageStatus>().unwrap(), MessageStatus::InProgress);
    }
}
