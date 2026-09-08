//! Process-level coverage of gateway-owned model startup and shutdown.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};

const TEST_TIMEOUT: Duration = Duration::from_secs(10);

struct ServerProcess {
    child: Child,
    directory: tempfile::TempDir,
}

impl ServerProcess {
    fn start(upstream_port: u16, extra_args: &[&str]) -> Self {
        let directory = tempfile::tempdir().expect("temporary server home");
        let python = directory.path().join("python");
        // exec preserves the child PID without spawning any grandchildren.
        fs::write(&python, "#!/bin/sh\nexec /bin/sleep 120\n").unwrap();
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755)).unwrap();
        let log = fs::File::create(directory.path().join("server.log")).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_agentic-server"))
            .env_clear()
            .env("PATH", directory.path())
            .env("AGENTIC_API_HOME", directory.path())
            .args([
                "serve",
                "mock-model",
                "--port",
                &upstream_port.to_string(),
                "--gateway-host",
                "127.0.0.1",
                "--gateway-port",
                "0",
                "--llm-ready-interval-s",
                "60",
            ])
            .args(extra_args)
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .expect("agentic-server must start");
        Self { child, directory }
    }

    fn log(&self) -> String {
        fs::read_to_string(self.directory.path().join("server.log")).unwrap()
    }

    fn recorded_model_pid(&self) -> Option<u32> {
        self.log().lines().find_map(|line| {
            line.split_once("spawned vLLM subprocess (pid ")?
                .1
                .split_once(')')?
                .0
                .parse()
                .ok()
        })
    }

    fn model_pid(&self) -> u32 {
        self.recorded_model_pid()
            .unwrap_or_else(|| panic!("model did not start: {}", self.log()))
    }

    async fn wait(&mut self) -> ExitStatus {
        timeout(TEST_TIMEOUT, async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    return status;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("server did not exit promptly: {}", self.log()))
    }

    async fn stop(&mut self, signal: &str) {
        let pid = self.model_pid();
        send_signal(self.child.id(), signal);
        let status = self.wait().await;
        assert!(!process_exists(pid), "owned model subprocess survived shutdown");
        assert!(status.success(), "shutdown failed: {status}; {}", self.log());
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Also clean up the model when a regression leaves it orphaned.
        if let Some(pid) = self.recorded_model_pid() {
            let _ = Command::new("kill").args(["-KILL", &pid.to_string()]).output();
        }
    }
}

fn process_exists(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .output()
        .expect("check subprocess existence")
        .status
        .success()
}

fn send_signal(pid: u32, signal: &str) {
    assert!(
        Command::new("kill")
            .args([signal, &pid.to_string()])
            .status()
            .expect("send signal")
            .success()
    );
}

async fn listener() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").await.unwrap()
}

async fn health_request(listener: &TcpListener) -> TcpStream {
    timeout(TEST_TIMEOUT, async {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while !request.ends_with(b"\r\n\r\n") {
            assert!(request.len() < 4096, "unexpected health request length");
            request.push(stream.read_u8().await.unwrap());
        }
        assert!(request.starts_with(b"GET /health HTTP/1.1\r\n"));
        stream
    })
    .await
    .expect("server must send a real readiness request")
}

async fn stop_during_readiness(signal: &str) {
    let upstream = listener().await;
    let mut server = ServerProcess::start(upstream.local_addr().unwrap().port(), &["--db-url", "sqlite::memory:"]);
    let _stalled_probe = health_request(&upstream).await;
    server.stop(signal).await;
}

#[tokio::test]
async fn sigterm_during_readiness_reaps_model() {
    stop_during_readiness("-TERM").await;
}

#[tokio::test]
async fn sigint_during_readiness_reaps_model() {
    stop_during_readiness("-INT").await;
}

async fn stalled_storage_startup() -> (ServerProcess, TcpStream) {
    let database = listener().await;
    let url = format!("postgres://test:test@{}/test", database.local_addr().unwrap());
    let server = ServerProcess::start(0, &["--skip-llm-ready-check", "--db-url", &url]);
    let (stream, _) = timeout(TEST_TIMEOUT, database.accept())
        .await
        .expect("server must attempt its configured database connection")
        .unwrap();
    (server, stream)
}

#[tokio::test]
async fn sigterm_during_storage_startup_reaps_model() {
    let (mut server, _stalled_database) = stalled_storage_startup().await;
    server.stop("-TERM").await;
}

#[tokio::test]
async fn sigint_during_storage_startup_reaps_model() {
    let (mut server, _stalled_database) = stalled_storage_startup().await;
    server.stop("-INT").await;
}

#[tokio::test]
async fn model_exit_during_storage_startup_is_reported_promptly() {
    let (mut server, _stalled_database) = stalled_storage_startup().await;
    let pid = server.model_pid();
    send_signal(pid, "-KILL");
    let status = server.wait().await;
    assert!(!status.success());
    assert!(server.log().contains("LlmProcessExited"), "{}", server.log());
    assert!(!process_exists(pid), "exited model must be reaped");
}

#[tokio::test]
async fn readiness_timeout_still_reaps_model_and_returns_error() {
    let upstream = listener().await;
    let mut server = ServerProcess::start(
        upstream.local_addr().unwrap().port(),
        &["--llm-ready-timeout-s", "1", "--db-url", "sqlite::memory:"],
    );
    let _stalled_probe = health_request(&upstream).await;
    let pid = server.model_pid();
    assert!(!server.wait().await.success());
    assert!(server.log().contains("LlmTimeout"), "{}", server.log());
    assert!(!process_exists(pid), "timed-out model must be reaped");
}

#[tokio::test]
async fn storage_startup_failure_still_reaps_model_and_returns_error() {
    let directory = tempfile::tempdir().unwrap();
    let url = format!("sqlite://{}", directory.path().display());
    let mut server = ServerProcess::start(0, &["--skip-llm-ready-check", "--db-url", &url]);
    assert!(!server.wait().await.success());
    assert!(server.log().contains("Error:"), "{}", server.log());
    assert!(!process_exists(server.model_pid()));
}

#[tokio::test]
async fn sigterm_after_gateway_startup_still_reaps_model() {
    let upstream = listener().await;
    let mut server = ServerProcess::start(upstream.local_addr().unwrap().port(), &["--db-url", "sqlite::memory:"]);
    health_request(&upstream)
        .await
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    timeout(TEST_TIMEOUT, async {
        while !server.log().contains("gateway listening on") {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("gateway did not start: {}", server.log()));
    server.stop("-TERM").await;
}
