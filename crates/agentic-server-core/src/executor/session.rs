//! Transient continuation state owned by one serial response session.

#[cfg(test)]
#[path = "session_budget_tests.rs"]
mod budget_tests;

use std::fmt;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use serde::Serialize;
use tokio::sync::Notify;

use super::{ExecutorError, ExecutorResult};
use crate::storage::{InOutItem, ResponseMetadata};
use crate::types::io::{InputItem, OutputItem};
use crate::utils::common::serialized_size_up_to;

/// One latest response checkpoint with a lifetime independent of durable storage.
///
/// A session admits one active turn. The owner must outlive execution; dropping it
/// clears cached state and prevents a late completion from publishing another
/// checkpoint. Active turns must still be cancelled/joined by their caller.
/// Budgets constrain retained item count and serialized size, not total heap use.
#[derive(Debug)]
pub struct ResponseSession {
    state: Arc<Mutex<SessionState>>,
    idle: Arc<Notify>,
    group: Option<Arc<Mutex<SessionGroupState>>>,
    budget: Option<Arc<CheckpointBudget>>,
}

#[derive(Debug)]
struct SessionState {
    latest: Option<Arc<RetainedCheckpoint>>,
    active: bool,
    closed: bool,
    max_items: NonZeroUsize,
    max_bytes: NonZeroUsize,
}

/// A bounded set of independent serial sessions sharing cached parent lookup.
///
/// The caller owns routing and scheduling. Create one member per logical session
/// and keep it while idle; queue disposal must not discard its retained state.
/// The group caps sessions created over its entire lifetime, not only active
/// sessions. Every member inherits the same item and serialized-byte budgets.
/// An aggregate serialized-byte budget covers cached checkpoints, pinned parents
/// and prepared replacements (including those awaiting durable persistence).
/// Sharing the same immutable parent counts it once. Replacement needs headroom
/// for old and new state until publication succeeds; no credit is granted for a
/// still-live parent. Executor input copies and temporary construction allocations
/// are outside this retention budget. Callers must separately bound active work;
/// this is not a heap-memory limit.
///
/// Dropping the group invalidates every member even if its handle survives.
/// Callers must still cancel and join active execution tasks.
#[derive(Debug)]
pub struct ResponseSessionGroup {
    state: Arc<Mutex<SessionGroupState>>,
    max_sessions: NonZeroUsize,
    max_items: NonZeroUsize,
    max_bytes: NonZeroUsize,
    budget: Arc<CheckpointBudget>,
}

#[derive(Debug)]
struct SessionGroupState {
    closed: bool,
    members: Vec<Weak<Mutex<SessionState>>>,
}

impl ResponseSessionGroup {
    #[must_use]
    pub fn new(
        max_sessions: NonZeroUsize,
        max_items: NonZeroUsize,
        max_bytes: NonZeroUsize,
        max_retained_bytes: NonZeroUsize,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionGroupState {
                closed: false,
                members: Vec::new(),
            })),
            max_sessions,
            max_items,
            max_bytes,
            budget: Arc::new(CheckpointBudget {
                used: AtomicUsize::new(0),
                limit: max_retained_bytes,
            }),
        }
    }

    /// Create another independent member without retaining its handle here.
    ///
    /// # Errors
    ///
    /// Returns an invalid-request error at the lifetime session cap or after
    /// closure, and a stream error if the group state lock is poisoned.
    pub fn new_session(&self) -> ExecutorResult<ResponseSession> {
        let mut state = lock_group(&self.state)?;
        if state.closed || state.members.len() >= self.max_sessions.get() {
            return Err(ExecutorError::InvalidRequest(
                "response session group is closed or has reached its session limit".to_owned(),
            ));
        }
        let mut session = ResponseSession::new(self.max_items, self.max_bytes);
        session.group = Some(Arc::clone(&self.state));
        session.budget = Some(Arc::clone(&self.budget));
        state.members.push(Arc::downgrade(&session.state));
        Ok(session)
    }
}

impl Drop for ResponseSessionGroup {
    fn drop(&mut self) {
        let members = {
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            state.closed = true;
            std::mem::take(&mut state.members)
        };
        // Never hold the group lock while closing members. A begin operation
        // takes the group lock before looking at any member, with no await.
        for member in members {
            if let Some(member) = member.upgrade() {
                let mut state = member.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                state.closed = true;
                state.latest = None;
            }
        }
    }
}

impl ResponseSession {
    #[must_use]
    pub fn new(max_items: NonZeroUsize, max_bytes: NonZeroUsize) -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionState {
                latest: None,
                active: false,
                closed: false,
                max_items,
                max_bytes,
            })),
            idle: Arc::new(Notify::new()),
            group: None,
            budget: None,
        }
    }

    /// Wait until the active turn has published or dropped its continuation.
    ///
    /// This is not cancellation: callers must first drop the execution stream
    /// or otherwise stop its worker. It prevents a subsequent serial request
    /// racing the asynchronous disposal triggered by dropping that stream.
    /// The ending lease releases its parent reference before becoming idle;
    /// other sessions may still keep that shared checkpoint alive and charged.
    ///
    /// # Errors
    ///
    /// Returns a stream error if the session state lock is poisoned.
    pub async fn wait_until_idle(&self) -> ExecutorResult<()> {
        loop {
            let notified = self.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !lock_state(&self.state)?.active {
                return Ok(());
            }
            notified.await;
        }
    }

    /// Discard a referenced checkpoint when a request fails before execution.
    ///
    /// An unrelated checkpoint is preserved. This does not cancel an active turn
    /// or release its execution slot; the caller still owns that turn's lifetime.
    ///
    /// # Errors
    ///
    /// Returns a stream error if the session state lock is poisoned.
    pub fn discard_cached_response(&self, response_id: &str) -> ExecutorResult<()> {
        let mut state = lock_state(&self.state)?;
        if state
            .latest
            .as_ref()
            .is_some_and(|latest| latest.response_id == response_id)
        {
            state.latest = None;
        }
        Ok(())
    }

    pub(crate) fn begin(&self, previous_id: Option<&str>) -> ExecutorResult<ResponseContinuation> {
        let group = self.group.as_ref().map(|group| lock_group(group)).transpose()?;
        if group.as_ref().is_some_and(|group| group.closed) {
            return Err(ExecutorError::InvalidRequest(
                "response session group has closed".to_owned(),
            ));
        }
        let mut state = lock_state(&self.state)?;
        if state.closed || state.active {
            return Err(ExecutorError::InvalidRequest(
                "response session is closed or already executing a turn".to_owned(),
            ));
        }
        let parent = state
            .latest
            .as_ref()
            .filter(|entry| Some(entry.response_id.as_str()) == previous_id)
            .cloned();
        state.active = true;
        let mut continuation = ResponseContinuation {
            state: Arc::clone(&self.state),
            idle: Arc::clone(&self.idle),
            parent,
            budget: self.budget.clone(),
            history_replaced: false,
            recorded_output_count: 0,
            finished: false,
        };
        drop(state);
        // Resolve a fork only when execution begins, not when its request queues.
        // Never hold two member locks. Once cloned, the immutable parent remains
        // pinned even if its source session advances or fails.
        if let (None, Some(previous_id), Some(group)) = (continuation.parent.as_ref(), previous_id, group.as_ref()) {
            for member in group.members.iter().filter_map(Weak::upgrade) {
                if Arc::ptr_eq(&member, &self.state) {
                    continue;
                }
                let source = lock_state(&member)?;
                if source.closed {
                    continue;
                }
                if let Some(parent) = source
                    .latest
                    .as_ref()
                    .filter(|parent| parent.response_id == previous_id)
                {
                    continuation.parent = Some(Arc::clone(parent));
                    break;
                }
            }
        }
        Ok(continuation)
    }
}

impl Drop for ResponseSession {
    fn drop(&mut self) {
        // Recover a poisoned guard only for disposal, never for reuse.
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        state.latest = None;
    }
}

/// An active turn's coherent parent snapshot and completion lease.
///
/// Constructed by the executor, not serialized into split-execution contexts.
/// Dropping an unfinished lease evicts its referenced cached parent and releases
/// the serial execution slot, including on rehydration, tool or stream errors.
#[derive(Debug)]
pub struct ResponseContinuation {
    state: Arc<Mutex<SessionState>>,
    idle: Arc<Notify>,
    pub(crate) parent: Option<Arc<RetainedCheckpoint>>,
    budget: Option<Arc<CheckpointBudget>>,
    history_replaced: bool,
    recorded_output_count: usize,
    finished: bool,
}

#[derive(Serialize)]
pub(crate) struct ResponseCheckpoint {
    pub response_id: String,
    pub conversation_id: Option<String>,
    pub history: Vec<InputItem>,
    pub metadata: ResponseMetadata,
    pub durable: bool,
}

impl fmt::Debug for ResponseCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResponseCheckpoint")
            .field("items", &self.history.len())
            .finish_non_exhaustive()
    }
}

/// An immutable checkpoint paired with its lifetime charge. Keeping the fields
/// private and omitting `DerefMut` prevents post-reservation size changes.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub(crate) struct RetainedCheckpoint {
    checkpoint: ResponseCheckpoint,
    #[serde(skip)]
    _reservation: Option<CheckpointReservation>,
}

impl Deref for RetainedCheckpoint {
    type Target = ResponseCheckpoint;

    fn deref(&self) -> &Self::Target {
        &self.checkpoint
    }
}

#[derive(Debug)]
struct CheckpointBudget {
    used: AtomicUsize,
    limit: NonZeroUsize,
}

impl CheckpointBudget {
    fn reserve(self: &Arc<Self>, bytes: usize) -> ExecutorResult<CheckpointReservation> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|total| *total <= self.limit.get())
            })
            .map_err(|_| aggregate_budget_error())?;
        Ok(CheckpointReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }
}

#[derive(Debug)]
struct CheckpointReservation {
    budget: Arc<CheckpointBudget>,
    bytes: usize,
}

impl Drop for CheckpointReservation {
    fn drop(&mut self) {
        let previous = self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
        debug_assert!(previous >= self.bytes, "checkpoint charge released exactly once");
    }
}

fn aggregate_budget_error() -> ExecutorError {
    ExecutorError::PayloadTooLarge(
        "response continuation exceeds the aggregate retained checkpoint budget; release inactive state or replay a compacted input window"
            .to_owned(),
    )
}

impl ResponseContinuation {
    /// The executor has replaced this turn's input with a canonical compacted window.
    pub(crate) fn mark_history_replaced(&mut self) {
        self.history_replaced = true;
    }

    /// The loop has recorded these outputs in canonical inference-round order.
    /// Public output remains complete, but persistence must not append it again.
    pub(crate) fn mark_outputs_recorded(&mut self, output_count: usize) {
        self.recorded_output_count = output_count;
    }

    pub(crate) fn retains_output(&self, index: usize, item: &OutputItem) -> bool {
        index >= self.recorded_output_count || matches!(item, OutputItem::McpListTools(_))
    }

    /// Retain orchestration records across compaction without restoring the old
    /// model context. MCP discovery still needs to know which servers were listed.
    pub(crate) fn parent_items(&self) -> impl Iterator<Item = &InputItem> {
        self.parent
            .iter()
            .flat_map(|parent| parent.history.iter())
            .filter(|item| !self.history_replaced || matches!(item, InputItem::McpListTools(_)))
    }

    pub(crate) fn checkpoint(
        &self,
        response_id: String,
        conversation_id: Option<String>,
        metadata: &ResponseMetadata,
        new_items: &[InOutItem],
        durable: bool,
    ) -> ExecutorResult<RetainedCheckpoint> {
        let mut history = self.parent_items().cloned().collect::<Vec<_>>();
        history.extend(InOutItem::into_input_items(new_items.to_vec()));
        let mut metadata = metadata.clone();
        if let Some(tools) = metadata.effective_tools.as_mut() {
            for tool in tools {
                tool.sanitize_for_persistence();
            }
        }
        let checkpoint = ResponseCheckpoint {
            response_id,
            conversation_id,
            history,
            metadata,
            durable,
        };
        let (max_items, max_bytes) = {
            let state = lock_state(&self.state)?;
            (state.max_items.get(), state.max_bytes.get())
        };
        let bytes = if checkpoint.history.len() <= max_items {
            serialized_size_up_to(&checkpoint, max_bytes)?
        } else {
            None
        };
        let Some(bytes) = bytes else {
            return Err(ExecutorError::PayloadTooLarge(
                "response continuation exceeds the session checkpoint budget; replay a compacted input window"
                    .to_owned(),
            ));
        };
        self.retain(checkpoint, bytes)
    }

    /// A durable fallback becomes a live pinned parent before inference. It must
    /// share the aggregate budget instead of bypassing it through storage.
    pub(crate) fn retain_parent(&self, checkpoint: ResponseCheckpoint) -> ExecutorResult<RetainedCheckpoint> {
        let bytes = if let Some(budget) = &self.budget {
            serialized_size_up_to(&checkpoint, budget.limit.get())?.ok_or_else(aggregate_budget_error)?
        } else {
            0 // Standalone sessions retain their existing per-completion policy.
        };
        self.retain(checkpoint, bytes)
    }

    fn retain(&self, checkpoint: ResponseCheckpoint, bytes: usize) -> ExecutorResult<RetainedCheckpoint> {
        let reservation = self.budget.as_ref().map(|budget| budget.reserve(bytes)).transpose()?;
        Ok(RetainedCheckpoint {
            checkpoint,
            _reservation: reservation,
        })
    }

    pub(crate) fn publish(mut self, checkpoint: RetainedCheckpoint) -> ExecutorResult<()> {
        {
            let mut state = lock_state(&self.state)?;
            if state.closed {
                return Err(ExecutorError::InvalidRequest("response session has closed".to_owned()));
            }
            state.latest = Some(Arc::new(checkpoint));
            // Return this lease's parent charge before another turn can observe
            // an idle slot, including a waiter resumed by notify_waiters below.
            self.parent = None;
            state.active = false;
        }
        self.finished = true;
        self.idle.notify_waiters();
        Ok(())
    }
}

impl Drop for ResponseContinuation {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.parent.as_ref().is_some_and(|parent| {
            state
                .latest
                .as_ref()
                .is_some_and(|latest| latest.response_id == parent.response_id)
        }) {
            state.latest = None;
        }
        // Field destructors run after Drop returns, too late for a waiter that
        // can immediately reuse the slot and reserve the released capacity.
        self.parent = None;
        state.active = false;
        drop(state);
        self.idle.notify_waiters();
    }
}

fn lock_state(state: &Mutex<SessionState>) -> ExecutorResult<MutexGuard<'_, SessionState>> {
    state
        .lock()
        .map_err(|_| ExecutorError::StreamError("response session state is unavailable".to_owned()))
}

fn lock_group(state: &Mutex<SessionGroupState>) -> ExecutorResult<MutexGuard<'_, SessionGroupState>> {
    state
        .lock()
        .map_err(|_| ExecutorError::StreamError("response session group state is unavailable".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::io::ResponsesInput;
    use serde_json::json;

    fn session(items: usize, bytes: usize) -> ResponseSession {
        ResponseSession::new(NonZeroUsize::new(items).unwrap(), NonZeroUsize::new(bytes).unwrap())
    }

    fn new_checkpoint(lease: &ResponseContinuation, id: &str) -> RetainedCheckpoint {
        let input = ResponsesInput::Text("private prompt".to_owned());
        let items = Vec::from(&input).into_iter().map(InOutItem::Input).collect::<Vec<_>>();
        lease
            .checkpoint(id.to_owned(), None, &ResponseMetadata::default(), &items, false)
            .unwrap()
    }

    fn complete(session: &ResponseSession, id: &str) {
        let lease = session.begin(None).unwrap();
        let checkpoint = new_checkpoint(&lease, id);
        lease.publish(checkpoint).unwrap();
    }

    fn group(count: usize) -> ResponseSessionGroup {
        ResponseSessionGroup::new(
            NonZeroUsize::new(count).unwrap(),
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(10_000).unwrap(),
            NonZeroUsize::new(100_000).unwrap(),
        )
    }

    #[test]
    fn group_session_cap_counts_idle_and_dropped_members() {
        let group = group(2);
        let first = group.new_session().unwrap();
        let second = group.new_session().unwrap();
        complete(&first, "resp_1");
        assert!(group.new_session().is_err());
        drop(second);
        assert!(group.new_session().is_err(), "the cap covers the entire group lifetime");
        assert!(first.begin(Some("resp_1")).unwrap().parent.is_some());
    }

    #[test]
    fn grouped_members_inherit_item_and_byte_budgets() {
        for (items, bytes) in [(1, 10_000), (10, 1)] {
            let group = ResponseSessionGroup::new(
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(items).unwrap(),
                NonZeroUsize::new(bytes).unwrap(),
                NonZeroUsize::new(100_000).unwrap(),
            );
            let member = group.new_session().unwrap();
            let lease = member.begin(None).unwrap();
            let input = Vec::from(&ResponsesInput::Text("private prompt".to_owned()));
            let input = input
                .iter()
                .cycle()
                .take(2)
                .cloned()
                .map(InOutItem::Input)
                .collect::<Vec<_>>();
            assert!(matches!(
                lease.checkpoint("resp_1".to_owned(), None, &ResponseMetadata::default(), &input, false),
                Err(ExecutorError::PayloadTooLarge(_))
            ));
        }
    }

    #[test]
    fn groups_do_not_share_parent_lookup() {
        let first_group = group(1);
        let second_group = group(1);
        let source = first_group.new_session().unwrap();
        let other = second_group.new_session().unwrap();
        complete(&source, "resp_1");
        assert!(other.begin(Some("resp_1")).unwrap().parent.is_none());
    }

    #[test]
    fn failed_fork_preserves_source_and_unrelated_destination_checkpoint() {
        let group = group(2);
        let source = group.new_session().unwrap();
        let target = group.new_session().unwrap();
        complete(&source, "resp_source");
        complete(&target, "resp_target");
        let fork = target.begin(Some("resp_source")).unwrap();
        assert!(fork.parent.is_some());
        drop(fork);
        assert!(source.begin(Some("resp_source")).unwrap().parent.is_some());
        assert!(target.begin(Some("resp_target")).unwrap().parent.is_some());
    }

    #[test]
    fn fork_cannot_find_parent_evicted_before_execution_starts() {
        let group = group(2);
        let source = group.new_session().unwrap();
        let target = group.new_session().unwrap();
        complete(&source, "resp_source");
        drop(source.begin(Some("resp_source")).unwrap());
        assert!(target.begin(Some("resp_source")).unwrap().parent.is_none());
    }

    #[test]
    fn dropped_group_invalidates_live_members_and_rejects_late_fork_publication() {
        let group = group(2);
        let source = group.new_session().unwrap();
        let target = group.new_session().unwrap();
        complete(&source, "resp_source");
        let fork = target.begin(Some("resp_source")).unwrap();
        let weak = Arc::downgrade(fork.parent.as_ref().unwrap());
        let checkpoint = new_checkpoint(&fork, "resp_fork");
        drop(group);
        assert!(source.begin(None).is_err());
        assert!(weak.upgrade().is_some(), "the active fork still pins its parent");
        assert!(fork.publish(checkpoint).is_err());
        assert!(weak.upgrade().is_none());
        assert!(target.begin(None).is_err());
    }

    #[test]
    fn group_does_not_keep_dropped_members_or_their_checkpoints_alive() {
        let group = group(2);
        let source = group.new_session().unwrap();
        let target = group.new_session().unwrap();
        complete(&source, "resp_source");
        let member = Arc::downgrade(&source.state);
        let checkpoint = Arc::downgrade(source.state.lock().unwrap().latest.as_ref().unwrap());
        drop(source);
        assert!(member.upgrade().is_none());
        assert!(checkpoint.upgrade().is_none());
        assert!(target.begin(Some("resp_source")).unwrap().parent.is_none());
    }

    #[test]
    fn session_rejects_concurrent_turns_and_releases_cancelled_lease() {
        let session = session(10, 10_000);
        let lease = session.begin(None).unwrap();
        assert!(session.begin(None).is_err());
        drop(lease);
        assert!(session.begin(None).is_ok());
    }

    #[tokio::test]
    async fn idle_wait_is_not_cancellation_and_is_notified_by_lease_disposal() {
        let session = session(10, 10_000);
        session.wait_until_idle().await.unwrap();
        let lease = session.begin(None).unwrap();
        let waiting = session.wait_until_idle();
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        assert!(session.begin(None).is_err());
        drop(lease);
        waiting.await.unwrap();
        assert!(session.begin(None).is_ok());
    }

    #[tokio::test]
    async fn successful_publication_wakes_all_idle_waiters() {
        let session = session(10, 10_000);
        let lease = session.begin(None).unwrap();
        let checkpoint = new_checkpoint(&lease, "resp_1");
        let first = session.wait_until_idle();
        let second = session.wait_until_idle();
        tokio::pin!(first, second);
        assert!(futures::poll!(&mut first).is_pending());
        assert!(futures::poll!(&mut second).is_pending());
        lease.publish(checkpoint).unwrap();
        first.await.unwrap();
        second.await.unwrap();
        assert!(session.begin(Some("resp_1")).unwrap().parent.is_some());
    }

    #[test]
    fn owner_drop_releases_checkpoint_and_rejects_late_publication() {
        let session = session(10, 10_000);
        complete(&session, "resp_1");
        let lease = session.begin(Some("resp_1")).unwrap();
        let weak = Arc::downgrade(lease.parent.as_ref().unwrap());
        let checkpoint = new_checkpoint(&lease, "resp_2");
        drop(session);
        assert!(lease.publish(checkpoint).is_err());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn successful_replacement_does_not_keep_ancestor_objects() {
        let session = session(10, 10_000);
        complete(&session, "resp_1");
        let lease = session.begin(Some("resp_1")).unwrap();
        let weak = Arc::downgrade(lease.parent.as_ref().unwrap());
        let checkpoint = new_checkpoint(&lease, "resp_2");
        assert_eq!(checkpoint.history.len(), 2);
        lease.publish(checkpoint).unwrap();
        assert!(weak.upgrade().is_none());
        let lease = session.begin(Some("resp_2")).unwrap();
        assert_eq!(lease.parent.as_ref().unwrap().history.len(), 2);
    }

    #[test]
    fn failed_continuation_evicts_parent_but_unknown_id_does_not() {
        let session = session(10, 10_000);
        complete(&session, "resp_1");
        drop(session.begin(Some("unknown")).unwrap());
        let lease = session.begin(Some("resp_1")).unwrap();
        assert!(lease.parent.is_some());
        drop(lease);
        assert!(session.begin(Some("resp_1")).unwrap().parent.is_none());
    }

    #[test]
    fn separate_sessions_have_no_shared_lookup() {
        let first = session(10, 10_000);
        let second = session(10, 10_000);
        complete(&first, "resp_1");
        assert!(second.begin(Some("resp_1")).unwrap().parent.is_none());
    }

    #[test]
    fn explicit_discard_preserves_the_active_snapshot_and_execution_slot() {
        let session = session(10, 10_000);
        complete(&session, "resp_1");
        session.discard_cached_response("unrelated").unwrap();
        let lease = session.begin(Some("resp_1")).unwrap();
        assert!(lease.parent.is_some());
        session.discard_cached_response("resp_1").unwrap();
        assert!(lease.parent.is_some(), "the active snapshot remains pinned");
        assert!(session.begin(None).is_err(), "discard is not cancellation");
        drop(lease);
        assert!(session.begin(Some("resp_1")).unwrap().parent.is_none());
    }

    #[test]
    fn item_budget_is_enforced_on_the_complete_history() {
        let session = session(1, 10_000);
        complete(&session, "resp_1");
        let lease = session.begin(Some("resp_1")).unwrap();
        let input = serde_json::from_value(json!({"type":"message", "role":"user", "content":"next"})).unwrap();
        assert!(matches!(
            lease.checkpoint(
                "resp_2".to_owned(),
                None,
                &ResponseMetadata::default(),
                &[InOutItem::Input(input)],
                false
            ),
            Err(ExecutorError::PayloadTooLarge(_))
        ));
    }

    #[test]
    fn byte_budget_includes_metadata_and_exact_boundary() {
        let source = session(10, 10_000);
        let lease = source.begin(None).unwrap();
        let checkpoint = new_checkpoint(&lease, "resp_1");
        let size = serialized_size_up_to(&checkpoint, usize::MAX).unwrap().unwrap();
        let exact = session(10, size);
        complete(&exact, "resp_1");
        let too_small = session(10, size - 1);
        let lease = too_small.begin(None).unwrap();
        let metadata = ResponseMetadata {
            effective_instructions: Some("large metadata".repeat(100)),
            ..ResponseMetadata::default()
        };
        assert!(matches!(
            lease.checkpoint("resp_1".to_owned(), None, &metadata, &[], false),
            Err(ExecutorError::PayloadTooLarge(_))
        ));
    }

    #[test]
    fn canonical_items_and_sanitized_mcp_metadata_are_retained() {
        let session = session(10, 10_000);
        let lease = session.begin(None).unwrap();
        let metadata = ResponseMetadata {
            effective_tools: Some(
                serde_json::from_value(json!([{
                    "type":"mcp", "server_label":"counter", "server_url":"https://example.com/mcp",
                    "headers":{"X-API-Key":"secret-header"}, "authorization":"secret-token", "require_approval":"never"
                }]))
                .unwrap(),
            ),
            ..ResponseMetadata::default()
        };
        let input = serde_json::from_value(json!({"type":"reasoning", "id":"rs_1",
            "content":[{"type":"reasoning_text", "text":"part one"},{"type":"reasoning_text", "text":"part two"}],
            "summary":[{"type":"summary_text", "text":"canonical summary"}], "encrypted_content":"opaque"
        }))
        .unwrap();
        let checkpoint = lease
            .checkpoint("resp_1".to_owned(), None, &metadata, &[InOutItem::Input(input)], false)
            .unwrap();
        let serialized = serde_json::to_value(&checkpoint).unwrap();
        assert_eq!(serialized["history"][0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(serialized["history"][0]["encrypted_content"], "opaque");
        assert!(!serialized.to_string().contains("secret-header"));
        assert!(!serialized.to_string().contains("secret-token"));
        assert!(!format!("{checkpoint:?}").contains("canonical summary"));
    }

    #[test]
    fn compaction_replacement_preserves_mcp_discovery_without_old_context() {
        let session = session(3, 10_000);
        let lease = session.begin(None).unwrap();
        let items = serde_json::from_value::<Vec<InputItem>>(json!([
            {"type":"message", "role":"user", "content":"private prompt"},
            {"type":"mcp_list_tools", "id":"mcp_1", "server_label":"counter", "tools":[]}
        ]))
        .unwrap()
        .into_iter()
        .map(InOutItem::Input)
        .collect::<Vec<_>>();
        let parent = lease
            .checkpoint("resp_1".to_owned(), None, &ResponseMetadata::default(), &items, false)
            .unwrap();
        lease.publish(parent).unwrap();
        let mut lease = session.begin(Some("resp_1")).unwrap();
        lease.mark_history_replaced();
        let compacted: Vec<InputItem> = serde_json::from_value(json!([
            {"type":"message", "role":"user", "id":"msg_kept", "status":"completed", "content":"retained user"},
            {"type":"compaction", "id":"cmp_1", "encrypted_content":"canonical summary"}
        ]))
        .unwrap();
        let items = compacted.into_iter().map(InOutItem::Input).collect::<Vec<_>>();
        let checkpoint = lease
            .checkpoint("resp_2".to_owned(), None, &ResponseMetadata::default(), &items, false)
            .unwrap();
        assert_eq!(checkpoint.history.len(), 3);
        assert!(matches!(checkpoint.history[0], InputItem::McpListTools(_)));
        assert!(matches!(checkpoint.history[2], InputItem::Compaction(_)));
        assert!(
            !serde_json::to_string(&checkpoint.history)
                .unwrap()
                .contains("private prompt")
        );
    }

    #[test]
    fn recorded_rounds_are_not_duplicated_and_mcp_discovery_is_retained() {
        let session = session(10, 10_000);
        let mut lease = session.begin(None).unwrap();
        let message: OutputItem = serde_json::from_value(json!({
            "type":"message", "id":"msg_1", "role":"assistant", "status":"completed", "content":[]
        }))
        .unwrap();
        let discovery: OutputItem = serde_json::from_value(json!({
            "type":"mcp_list_tools", "id":"mcp_1", "server_label":"counter", "tools":[]
        }))
        .unwrap();
        assert!(lease.retains_output(0, &message));
        lease.mark_outputs_recorded(2);
        assert!(!lease.retains_output(1, &message));
        assert!(lease.retains_output(2, &message));
        lease.mark_history_replaced();
        lease.mark_outputs_recorded(4);
        assert!(!lease.retains_output(2, &message));
        assert!(lease.retains_output(4, &message));
        assert!(lease.retains_output(0, &discovery));
    }
}
