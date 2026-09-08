use chrono::Utc;
use uuid::Uuid;

#[must_use]
pub fn uuid7_str(prefix: &str) -> String {
    format!("{}{}", prefix, Uuid::now_v7())
}

#[must_use]
pub fn utcnow_str() -> i64 {
    Utc::now().timestamp()
}

/// Serialize any type to JSON string.
///
/// Strict serialization - returns error if serialization fails.
/// Used in persistence operations where we control the data types.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails.
pub fn serialize_to_string<T: serde::Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string(value)
}

/// Count serialized JSON bytes without allocating the complete JSON buffer.
/// Returns `None` as soon as the serialization exceeds `limit`.
///
/// # Errors
/// Returns a serialization error unrelated to the size limit.
pub fn serialized_size_up_to<T: serde::Serialize>(value: &T, limit: usize) -> Result<Option<usize>, serde_json::Error> {
    struct Counter {
        used: usize,
        limit: usize,
        exceeded: bool,
    }
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.used) {
                self.exceeded = true;
                return Err(std::io::Error::other("serialized size limit exceeded"));
            }
            self.used += bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter {
        used: 0,
        limit,
        exceeded: false,
    };
    let result = serde_json::to_writer(&mut counter, value);
    if counter.exceeded {
        return Ok(None);
    }
    result?;
    Ok(Some(counter.used))
}

/// Serialize any type to a `serde_json::Value`.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails.
pub fn serialize_to_value<T: serde::Serialize>(value: &T) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::to_value(value)
}

/// Serializes `value` to a `serde_json::Value` and passes it to `then`,
/// logging `message` at `debug` level and returning `default` if
/// serialization fails.
///
/// Graceful serialization - used where callers fall back to a default result
/// rather than propagate a serialization error.
pub fn serialize_to_value_or_custom_default<T: serde::Serialize, R>(
    value: &T,
    message: &str,
    then: impl FnOnce(serde_json::Value) -> R,
    default: R,
) -> R {
    match serde_json::to_value(value) {
        Ok(config) => then(config),
        Err(error) => {
            tracing::debug!(error = %error, message);
            default
        }
    }
}

/// Deserialize JSON string to any type.
///
/// Strict deserialization - returns error if deserialization fails.
/// Used when we need explicit error handling for data integrity.
///
/// # Errors
///
/// Returns `serde_json::Error` if deserialization fails.
pub fn deserialize_from_str<T: serde::de::DeserializeOwned>(json_str: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(json_str)
}

/// Deserialize JSON string to any type with Default fallback.
///
/// Graceful deserialization - returns default value on error or empty string.
/// Used in read operations where we accept corrupted data gracefully.
#[must_use]
pub fn deserialize_from_str_or_default<T: serde::de::DeserializeOwned + Default>(json_str: &str) -> T {
    serde_json::from_str(json_str).unwrap_or_default()
}

/// Deserialize JSON string to any type, returning None on error.
///
/// Optional deserialization - returns None if JSON is invalid.
/// Convenience function for cases where None represents missing data.
#[must_use]
pub fn deserialize_from_str_opt<T: serde::de::DeserializeOwned>(json_str: &str) -> Option<T> {
    serde_json::from_str(json_str).ok()
}

/// Deserialize optional JSON String to any type, returning default on error or if None.
///
/// Graceful optional deserialization - returns default value for T if missing or invalid.
#[must_use]
pub fn deserialize_from_string_opt_or_default<T: serde::de::DeserializeOwned + Default>(
    json_str: &Option<String>,
) -> T {
    json_str
        .as_ref()
        .and_then(|s| deserialize_from_str_opt::<T>(s))
        .unwrap_or_default()
}

/// Deserialize optional JSON String to any type, returning None on error or if None.
///
/// Optional deserialization - returns None if missing or invalid JSON.
#[must_use]
pub fn deserialize_from_string_opt<T: serde::de::DeserializeOwned>(json_str: &Option<String>) -> Option<T> {
    json_str.as_ref().and_then(|s| deserialize_from_str_opt::<T>(s))
}

/// Deserialize a `serde_json::Value` into `T`.
///
/// # Errors
///
/// Returns `serde_json::Error` if the value's shape does not match `T`.
pub fn deserialize_from_value<T: serde::de::DeserializeOwned>(
    value: serde_json::Value,
) -> Result<T, serde_json::Error> {
    serde_json::from_value(value)
}

/// Deserializes a `serde_json::Value` and passes it to `then`, logging
/// `message` at `debug` level and returning `default` if deserialization fails.
///
/// Graceful deserialization - used where callers fall back to a default result
/// rather than propagate a deserialization error.
pub fn deserialize_from_value_or_custom_default<T: serde::de::DeserializeOwned, R>(
    value: serde_json::Value,
    message: &str,
    then: impl FnOnce(T) -> R,
    default: R,
) -> R {
    match serde_json::from_value(value) {
        Ok(value) => then(value),
        Err(error) => {
            tracing::debug!(error = %error, message);
            default
        }
    }
}

/// Deserialize a `serde_json::Value` into `T`, returning `None` on type mismatch.
#[must_use]
pub fn deserialize_from_value_opt<T: serde::de::DeserializeOwned>(value: serde_json::Value) -> Option<T> {
    serde_json::from_value(value).ok()
}

/// Serialize any type to JSON bytes, returning an empty `Vec` on error.
#[must_use]
pub fn serialize_to_vec_or_default<T: serde::Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}
