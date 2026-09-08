pub mod normalize;
pub mod types;
mod validate;

pub(crate) use normalize::is_data_frame;
pub use normalize::normalize_sse_line;
pub use types::{EventFrame, EventPayload, SSEEventType, SSEItemType, WireEvent};
pub(crate) use validate::{ValidatedFrame, ensure_supported_output_item_type, output_item_identity, validate_frame};
