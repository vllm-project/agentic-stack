//! Agentic executor.

pub mod accumulator;
pub mod compaction;
pub mod engine;
pub mod error;
pub mod function_sse;
pub mod inference;
pub mod messages_context;
pub mod messages_loop;
mod messages_request;
pub mod messages_stream;
pub mod modes;
pub mod persist;
pub mod rehydrate;
pub mod request;

mod gateway;
pub mod gateway_accumulator;
mod pending_calls;
mod upstream;

pub use compaction::compact_response;
pub use engine::{BoxStream, ExecuteRequest, create_conversation, execute};
pub use error::{ExecutorError, ExecutorResult};
pub use inference::call_inference;
pub use messages_context::MessagesRequestContext;
pub use messages_loop::{MessagesResponse, MessagesUpstream, run_messages_loop};
pub use messages_request::normalize_native_web_search_for_upstream;
pub use messages_stream::run_messages_stream;
pub use modes::{ConversationHandler, ResponseHandler};
pub use persist::{commit, persist_response, persist_turn};
pub use rehydrate::rehydrate_conversation;
pub use request::ExecutionContext;
pub use request::RequestContext;
pub use upstream::{UpstreamBody, decode_upstream, upstream_request};
