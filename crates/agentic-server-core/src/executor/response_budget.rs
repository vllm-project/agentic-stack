use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::executor::error::{ExecutorError, ExecutorResult};

pub(super) const MAX_EXECUTOR_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone)]
pub(super) struct ExecutorResponseBudget {
    remaining: Arc<AtomicUsize>,
}

impl ExecutorResponseBudget {
    pub(super) fn new() -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(MAX_EXECUTOR_RESPONSE_BYTES)),
        }
    }

    pub(super) fn consume(&self, bytes: usize) -> ExecutorResult<()> {
        self.remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(bytes)
            })
            .map(|_| ())
            .map_err(|_| {
                ExecutorError::StreamError(format!(
                    "executor response budget exceeded {MAX_EXECUTOR_RESPONSE_BYTES} bytes"
                ))
            })
    }
}
