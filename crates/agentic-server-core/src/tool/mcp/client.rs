use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use http::header::{HeaderName, HeaderValue};
use rmcp::ClientHandler;
use rmcp::ServiceExt;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientInfo, ClientRequest, Implementation,
    InitializeRequestParams, ProtocolVersion, ServerResult, Tool,
};
use rmcp::service::{ClientInitializeError, PeerRequestOptions, RoleClient, RunningService, ServiceError};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp_reqwest as http_client;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use super::bounded_http::BoundedMcpHttpClient;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
const TOOL_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpOperation {
    Connect,
    ListTools,
    CallTool,
}

impl fmt::Display for McpOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect => f.write_str("connect"),
            Self::ListTools => f.write_str("tools/list"),
            Self::CallTool => f.write_str("tools/call"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("failed to spawn MCP stdio server")]
    SpawnStdio(#[source] std::io::Error),

    #[error("failed to connect to MCP server")]
    Connect(#[source] Box<ClientInitializeError>),

    #[error("failed to resolve MCP server host")]
    ResolveHost(#[source] std::io::Error),

    #[error("MCP server URL has no resolvable host")]
    UnresolvableHost,

    #[error("failed to build MCP HTTP client")]
    BuildHttpClient(#[source] http_client::Error),

    #[error("invalid MCP HTTP header name")]
    InvalidHeaderName(#[source] http::header::InvalidHeaderName),

    #[error("invalid MCP HTTP header value")]
    InvalidHeaderValue(#[source] http::header::InvalidHeaderValue),

    #[error("MCP operation failed during {operation}")]
    Operation {
        operation: McpOperation,
        #[source]
        source: ServiceError,
    },

    #[error("MCP operation timed out during {operation}")]
    Timeout { operation: McpOperation },

    #[error("MCP tool arguments must be a JSON object")]
    InvalidArguments,

    #[error("MCP server returned an unexpected response during {operation}")]
    UnexpectedResponse { operation: McpOperation },
}

#[derive(Clone)]
struct AgenticMcpClientHandler;

impl ClientHandler for AgenticMcpClientHandler {
    fn get_info(&self) -> ClientInfo {
        InitializeRequestParams::new(
            ClientCapabilities::default(),
            Implementation::new("agentic-api", env!("CARGO_PKG_VERSION")),
        )
        .with_protocol_version(ProtocolVersion::V_2025_06_18)
    }
}

pub struct McpClient {
    inner: Arc<RunningService<RoleClient, AgenticMcpClientHandler>>,
    tool_timeout: Duration,
}

impl McpClient {
    /// Connects to an MCP server over streamable HTTP.
    ///
    /// # Errors
    ///
    /// Returns an error if URL resolution, HTTP client or header construction,
    /// the initialization timeout, or the MCP handshake fails.
    pub async fn connect(server_url: &str, headers: Option<HashMap<String, String>>) -> Result<Self, McpError> {
        tokio::time::timeout(CONNECTION_TIMEOUT, Self::connect_streamable_http(server_url, headers))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::Connect,
            })?
    }

    async fn connect_streamable_http(
        server_url: &str,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self, McpError> {
        let http_client = pinned_http_client(server_url).await?;
        let mut config = StreamableHttpClientTransportConfig::with_uri(server_url.to_owned());
        if let Some(headers) = headers.filter(|headers| !headers.is_empty()) {
            let mut custom_headers = HashMap::with_capacity(headers.len());
            for (name, value) in headers {
                custom_headers.insert(
                    HeaderName::try_from(name).map_err(McpError::InvalidHeaderName)?,
                    HeaderValue::try_from(value).map_err(McpError::InvalidHeaderValue)?,
                );
            }
            config = config.custom_headers(custom_headers);
        }
        let transport = StreamableHttpClientTransport::with_client(BoundedMcpHttpClient::new(http_client), config);
        let service = AgenticMcpClientHandler
            .serve(transport)
            .await
            .map_err(|error| McpError::Connect(Box::new(error)))?;

        Ok(Self {
            inner: Arc::new(service),
            tool_timeout: TOOL_TIMEOUT,
        })
    }

    /// Spawns a local stdio MCP server and connects over stdin/stdout.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::SpawnStdio`] if the process cannot be spawned.
    /// Returns [`McpError::Connect`] if the MCP initialization handshake fails.
    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&str>,
    ) -> Result<Self, McpError> {
        let mut command_builder = Command::new(command);
        command_builder
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .args(args);

        if let Some(env) = env {
            command_builder.envs(env);
        }

        if let Some(cwd) = cwd {
            command_builder.current_dir(cwd);
        }

        let (transport, stderr) = TokioChildProcess::builder(command_builder)
            .spawn()
            .map_err(McpError::SpawnStdio)?;

        if let Some(stderr) = stderr {
            let command = command.to_owned();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                loop {
                    match reader.next_line().await {
                        Ok(Some(line)) => {
                            tracing::info!(mcp.command = %command, %line, "MCP server stderr");
                        }
                        Ok(None) => break,
                        Err(error) => {
                            tracing::warn!(
                                mcp.command = %command,
                                error = %error,
                                "failed to read MCP server stderr"
                            );
                            break;
                        }
                    }
                }
            });
        }

        let service = tokio::time::timeout(CONNECTION_TIMEOUT, AgenticMcpClientHandler.serve(transport))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::Connect,
            })?
            .map_err(|error| McpError::Connect(Box::new(error)))?;

        Ok(Self {
            inner: Arc::new(service),
            tool_timeout: TOOL_TIMEOUT,
        })
    }

    /// Lists tools exposed by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Timeout`] if `tools/list` exceeds the configured timeout.
    /// Returns [`McpError::Operation`] if the server rejects or fails the request.
    pub async fn list_tools(&self) -> Result<Vec<Tool>, McpError> {
        let result = tokio::time::timeout(self.tool_timeout, self.inner.list_tools(None))
            .await
            .map_err(|_| McpError::Timeout {
                operation: McpOperation::ListTools,
            })?
            .map_err(|source| McpError::Operation {
                operation: McpOperation::ListTools,
                source,
            })?;

        Ok(result.tools)
    }

    /// Calls a tool exposed by the connected MCP server.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::InvalidArguments`] if `arguments` is not a JSON object.
    /// Returns [`McpError::Timeout`] if `tools/call` exceeds the configured timeout.
    /// Returns [`McpError::Operation`] if the server rejects or fails the request.
    /// Returns [`McpError::UnexpectedResponse`] if the server returns another response kind.
    pub async fn call_tool(&self, name: &str, arguments: Option<Value>) -> Result<CallToolResult, McpError> {
        let arguments = match arguments {
            Some(Value::Object(map)) => Some(map),
            Some(_) => return Err(McpError::InvalidArguments),
            None => None,
        };

        let mut params = CallToolRequestParams::new(name.to_owned());
        params.arguments = arguments;

        let result = tokio::time::timeout(self.tool_timeout, async {
            self.inner
                .peer()
                .send_request_with_option(
                    ClientRequest::CallToolRequest(rmcp::model::CallToolRequest::new(params)),
                    PeerRequestOptions::no_options(),
                )
                .await?
                .await_response()
                .await
        })
        .await
        .map_err(|_| McpError::Timeout {
            operation: McpOperation::CallTool,
        })?
        .map_err(|source| McpError::Operation {
            operation: McpOperation::CallTool,
            source,
        })?;

        match result {
            ServerResult::CallToolResult(result) => Ok(result),
            _ => Err(McpError::UnexpectedResponse {
                operation: McpOperation::CallTool,
            }),
        }
    }
}

/// Build the HTTP client used by an MCP connection with DNS pinned to the
/// addresses resolved during connection setup. This prevents a hostname from
/// resolving to different addresses between URL validation and later requests.
async fn pinned_http_client(server_url: &str) -> Result<http_client::Client, McpError> {
    let url = http_client::Url::parse(server_url).map_err(|_| McpError::UnresolvableHost)?;
    let port = url.port_or_known_default().ok_or(McpError::UnresolvableHost)?;
    match url.host().ok_or(McpError::UnresolvableHost)? {
        url::Host::Domain(host) => {
            let addresses = tokio::net::lookup_host((host, port))
                .await
                .map_err(McpError::ResolveHost)?
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(McpError::UnresolvableHost);
            }
            http_client_for_addresses(host, &addresses)
        }
        url::Host::Ipv4(_) | url::Host::Ipv6(_) => http_client_for_literal_address(),
    }
}

fn http_client_for_addresses(host: &str, addresses: &[SocketAddr]) -> Result<http_client::Client, McpError> {
    http_client::Client::builder()
        .no_proxy()
        .redirect(http_client::redirect::Policy::none())
        .resolve_to_addrs(host, addresses)
        .build()
        .map_err(McpError::BuildHttpClient)
}

fn http_client_for_literal_address() -> Result<http_client::Client, McpError> {
    http_client::Client::builder()
        .no_proxy()
        .redirect(http_client::redirect::Policy::none())
        .build()
        .map_err(McpError::BuildHttpClient)
}

#[cfg(test)]
mod tests {
    use super::{McpError, http_client_for_addresses, pinned_http_client};
    use std::io::{Read, Write};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener as StdTcpListener};
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener as TokioTcpListener;

    const PROXY_CHILD_ENV: &str = "AGENTIC_MCP_PROXY_TEST_CHILD";
    const PROXY_TARGET_ENV: &str = "AGENTIC_MCP_PROXY_TEST_TARGET";

    async fn spawn_http_server(response: &'static str) -> SocketAddr {
        let listener = TokioTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        address
    }

    #[tokio::test]
    async fn pinned_client_tries_every_resolved_address() {
        let reachable = spawn_http_server("HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await;
        let unreachable = SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], reachable.port()));
        let client = http_client_for_addresses("mcp.test", &[unreachable, reachable]).unwrap();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.get(format!("http://mcp.test:{}/", reachable.port())).send(),
        )
        .await
        .expect("client must try the reachable address")
        .unwrap();

        assert_eq!(response.status(), http::StatusCode::OK);
    }

    #[tokio::test]
    async fn pinned_client_does_not_follow_redirects() {
        let address = spawn_http_server(
            "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await;
        let client = http_client_for_addresses("mcp.test", &[address]).unwrap();

        let response = client
            .get(format!("http://mcp.test:{}/", address.port()))
            .send()
            .await
            .unwrap();

        assert!(response.status().is_redirection());
    }

    #[tokio::test]
    async fn pinned_client_rejects_malformed_urls() {
        assert!(matches!(
            pinned_http_client("not a URL").await,
            Err(McpError::UnresolvableHost)
        ));
    }

    #[tokio::test]
    async fn pinned_client_accepts_ipv6_literal_urls_without_dns() {
        pinned_http_client("http://[::1]:8000/mcp").await.unwrap();
    }

    fn serve_once(listener: &StdTcpListener, status: &str, accepted: &AtomicBool) {
        listener.set_nonblocking(true).unwrap();
        for _ in 0..200 {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    accepted.store(true, Ordering::SeqCst);
                    stream.set_nonblocking(false).unwrap();
                    let mut request = [0_u8; 1024];
                    let _ = stream.read(&mut request).unwrap();
                    write!(
                        stream,
                        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .unwrap();
                    return;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("test server failed: {error}"),
            }
        }
    }

    #[test]
    fn pinned_client_ignores_system_proxy() {
        if std::env::var_os(PROXY_CHILD_ENV).is_some() {
            let target = std::env::var(PROXY_TARGET_ENV).unwrap().parse::<SocketAddr>().unwrap();
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                let client = http_client_for_addresses("mcp.test", &[target]).unwrap();
                let response = client
                    .get(format!("http://mcp.test:{}/", target.port()))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), http::StatusCode::OK);
            });
            return;
        }

        let target_listener = StdTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let target_address = target_listener.local_addr().unwrap();
        let target_accepted = Arc::new(AtomicBool::new(false));
        let target_thread = {
            let accepted = Arc::clone(&target_accepted);
            thread::spawn(move || serve_once(&target_listener, "200 OK", &accepted))
        };

        let proxy_listener = StdTcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let proxy_address = proxy_listener.local_addr().unwrap();
        let proxy_accepted = Arc::new(AtomicBool::new(false));
        let proxy_thread = {
            let accepted = Arc::clone(&proxy_accepted);
            thread::spawn(move || serve_once(&proxy_listener, "418 I'm a teapot", &accepted))
        };

        let output = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("tool::mcp::client::tests::pinned_client_ignores_system_proxy")
            .arg("--nocapture")
            .env(PROXY_CHILD_ENV, "1")
            .env(PROXY_TARGET_ENV, target_address.to_string())
            .env("HTTP_PROXY", format!("http://{proxy_address}"))
            .env("http_proxy", format!("http://{proxy_address}"))
            .env("ALL_PROXY", format!("http://{proxy_address}"))
            .env("all_proxy", format!("http://{proxy_address}"))
            .env_remove("NO_PROXY")
            .env_remove("no_proxy")
            .output()
            .unwrap();

        target_thread.join().unwrap();
        proxy_thread.join().unwrap();
        assert!(
            output.status.success(),
            "child test failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(target_accepted.load(Ordering::SeqCst));
        assert!(!proxy_accepted.load(Ordering::SeqCst));
    }
}
