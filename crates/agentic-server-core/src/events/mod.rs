pub mod normalize;
pub mod types;

pub use normalize::normalize_sse_line;
pub(crate) use normalize::normalize_sse_value;
pub use types::{EventFrame, EventPayload, SSEEventType, SSEItemType, WireEvent};
