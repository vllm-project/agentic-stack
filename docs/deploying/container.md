# Run agentic-api in a container

The production image contains only the Rust gateway and its runtime libraries. It does not contain Python, vLLM, GPU libraries, model weights, Cargo, or the Rust toolchain. Run inference and PostgreSQL as external services.

## Install from crates.io instead

Since 0.4.0 the gateway is also published to crates.io, so a container is not required on a host with a Rust
toolchain (MSRV 1.85):

```console
cargo install agentic-server --version 0.4.0 --locked
```

This installs two binaries: `agentic`, the harness CLI (`agentic serve` starts the gateway alone; `agentic run
claude|codex` starts it and launches a harness), and `agentic-server`, the standalone server configured through the
same environment variables as the container below. The container remains the recommended path for Kubernetes and
for hosts without a toolchain.

## Build the image

The multi-stage build pins its Rust and Debian bases by digest, uses BuildKit caches, and copies only `agentic-server` into the runtime stage. Dependabot proposes weekly digest updates so base-image changes remain explicit and reviewable.

```console
DOCKER_BUILDKIT=1 docker build \
  --build-arg OCI_CREATED="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --build-arg OCI_REVISION="$(git rev-parse HEAD)" \
  --build-arg OCI_VERSION="dev" \
  --tag agentic-api:dev \
  .
```

For a release build, pass the version being packaged instead of `dev` and tag the image to match, for example
`OCI_VERSION="0.4.0"` with `--tag agentic-api:0.4.0`. That version tag (or, preferably, the digest recorded when
the image is pushed) is what an environment overlay such as the one in the
[Kubernetes guide](kubernetes.md) should pin.

CI also records the workflow name and run URL in the vLLM-compatible image labels. Local builds use `local` as the pipeline label unless `OCI_BUILD_PIPELINE` is supplied as a build argument.

Pass `--build-arg CARGO_BUILD_JOBS=<n>` if the builder needs a different concurrency limit. For multi-platform publishing, use a Buildx builder with native workers where available; emulation is slower.

## Configure the gateway

The image starts `agentic-server` in standalone mode. At minimum, set `LLM_API_BASE` to an external OpenAI-compatible or vLLM endpoint. Production deployments should also set `DATABASE_URL` to an external PostgreSQL database.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_API_BASE` | none | Required upstream inference URL |
| `GATEWAY_HOST` | `0.0.0.0` | Listen address |
| `GATEWAY_PORT` | `9000` | Listen port |
| `DATABASE_URL` | `$AGENTIC_API_HOME/agentic_api.db` | SQLite or PostgreSQL persistence URL |
| `AGENTIC_API_HOME` | `/var/lib/agentic-api` | User configuration and default local-state directory |
| `AGENTIC_MAX_REQUEST_BODY_SIZE_BYTES` | `10485760` | Maximum serialized request size for HTTP bodies and WebSocket messages; must be greater than zero |
| `POSTGRES_MAX_CONNECTIONS` | `10` | Maximum PostgreSQL connections per gateway replica |
| `POSTGRES_ACQUIRE_TIMEOUT_SECONDS` | `30` | Maximum wait for a PostgreSQL pool connection |
| `POSTGRES_LOCK_TIMEOUT_SECONDS` | `5` | Maximum wait for a PostgreSQL row or table lock |
| `POSTGRES_MIGRATION_TIMEOUT_SECONDS` | `300` | Lock and statement timeout while running startup migrations |
| `POSTGRES_STATEMENT_TIMEOUT_SECONDS` | `30` | Maximum runtime for normal PostgreSQL statements |
| `POSTGRES_IDLE_TIMEOUT_SECONDS` | `600` | Recycle idle PostgreSQL connections; `0` disables |
| `POSTGRES_MAX_LIFETIME_SECONDS` | `1800` | Recycle PostgreSQL connections after this lifetime; `0` disables |
| `OPENAI_API_KEY` | none | Credential sent to the upstream service when the client does not supply one |
| `OIDC_ISSUER` | none | Optional OIDC issuer for inbound bearer-token authentication |
| `OIDC_AUDIENCE` | none | Required token audience when `OIDC_ISSUER` is set |
| `SKIP_LLM_READY_CHECK` | `false` | Skip startup and recurring upstream probes for providers without `/health`; PostgreSQL readiness remains enforced |
| `CORS_ALLOWED_ORIGINS` | none | Comma-separated browser origins |

The container entrypoint rejects percent-encoded SQLite paths because SQLx decodes them before opening the database. Use a literal filesystem path or PostgreSQL instead.

Do not put credentials into the image or Docker build arguments. Inject them at runtime through a secret manager.

```console
docker run --rm --name agentic-api \
  --publish 127.0.0.1:9000:9000 \
  --env LLM_API_BASE=https://vllm.example.com \
  --env DATABASE_URL=postgresql://agentic-api@postgres.example.com/agentic_api \
  --env OPENAI_API_KEY \
  agentic-api:dev
```

Inbound authentication is disabled by default. `OPENAI_API_KEY` is an upstream credential, not a password for callers,
so keep the port bound to loopback unless an authenticated ingress protects it or OIDC is enabled.

To enable OIDC bearer authentication, configure both variables:

```console
docker run --rm --name agentic-api \
  --publish 127.0.0.1:9000:9000 \
  --env LLM_API_BASE=https://vllm.example.com \
  --env OPENAI_API_KEY \
  --env OIDC_ISSUER=https://identity.example.com \
  --env OIDC_AUDIENCE=agentic-api \
  agentic-api:dev
```

The gateway discovers the provider and its JSON Web Key Set before listening. `/health` and `/ready` remain public;
all `/v1/*` routes then require an OIDC `Authorization: Bearer` token. The identity token is consumed by the gateway,
and `OPENAI_API_KEY` supplies the upstream inference credential. See
[OIDC bearer authentication](../design/oidc-bearer-authentication.md) for the validation and key-rotation contract.
For a runnable GitHub-backed setup, follow [GitHub authentication with Dex](github-oidc.md).

If the upstream is running on the Docker host, use `http://host.docker.internal:<port>` on Docker Desktop. On Linux, add `--add-host host.docker.internal:host-gateway`.

### PostgreSQL production settings

Use the TLS settings required by the managed database provider. Prefer certificate and hostname verification when the provider supplies a CA certificate:

```console
DATABASE_URL='postgresql://agentic-api:password@postgres.example.com/agentic_api?sslmode=verify-full&sslrootcert=/run/secrets/postgres-ca.pem'
```

`sslmode=require` encrypts the connection but does not verify the server hostname. Mount private CA certificates and client keys from runtime secrets; do not copy them into the image.

`AGENTIC_MAX_REQUEST_BODY_SIZE_BYTES` bounds the encoded request the gateway will read — JSON overhead, replayed
conversation history, and base64 image attachments included. It is unrelated to the upstream model's token context
limit, so raising it does not raise what the inference server will accept. Because inline base64 attachments cost
roughly a third more than the source image, deployments that replay image-bearing conversations may need more than the
10 MiB default. Requests above the limit receive `413 Payload Too Large`; oversized WebSocket messages and frames are
rejected by the transport, which closes the connection. The `--max-request-body-size-bytes` argument takes precedence
over this variable, which in turn overrides `server.max_request_body_size_bytes` in the configuration file.

Size the pool across the whole deployment, not one process. Keep `replicas * POSTGRES_MAX_CONNECTIONS` below the managed database connection limit, with capacity reserved for migrations, administration, and failover. The acquire timeout bounds how long a request waits when the pool is exhausted. Lock and statement timeouts prevent a stalled transaction, slow query, or hot conversation from holding a connection indefinitely. Idle and lifetime recycling protect against stale connections and can be disabled with `0` only when the provider recommends it.

Connect directly to PostgreSQL or use a session-pooling proxy. Transaction-pooling proxies are unsupported because the
gateway configures `search_path`, lock timeouts, and statement timeouts for the lifetime of each pooled session.

Each gateway replica runs the embedded SQLx migrations during startup. PostgreSQL advisory locking serializes concurrent migration attempts, so replica startup is repeatable and only proceeds after the schema is ready. Migration lock waits and statements use the finite migration timeout rather than the shorter application lock timeout. A migration failure or timeout prevents that replica from serving traffic.

The gateway pins every PostgreSQL pool connection to the single schema on `search_path` that already contains the
persistence tables or SQLx migration history. This prevents an earlier empty schema from receiving shadow tables or
capturing runtime queries. Startup fails instead of guessing when persistence objects exist in more than one
`search_path` schema. Do not grant untrusted roles `CREATE` on any schema in the gateway role's `search_path`.

The first PostgreSQL startup on this release widens timestamp and sequence columns in place to 64-bit integers. PostgreSQL takes exclusive table locks for these changes, which prevents concurrent writes from being lost but can briefly pause older replicas. It also rebuilds the conversation sequence index as unique, so duplicate non-null `(conversation_id, seq)` values stop the migration instead of leaving conversation order ambiguous. Check for duplicates before the upgrade and resolve them according to the deployment's data-retention policy:

```sql
SELECT conversation_id, seq, COUNT(*)
FROM items
WHERE conversation_id IS NOT NULL AND seq IS NOT NULL
GROUP BY conversation_id, seq
HAVING COUNT(*) > 1;
```

For an existing large database, schedule the first upgraded replica during a maintenance window and set the migration timeout for the expected lock, rewrite, and index duration.

Drain replicas running an older release before enabling writes through this release. Older replicas do not take the per-conversation row lock and can allocate duplicate sequence numbers if they write alongside upgraded replicas.

Stored requests now fail if their response or conversation state cannot be persisted. For streaming requests, the gateway sends an error event instead of `response.completed`. Most client responses use the generic message `failed to persist response`; the underlying database error is written only to gateway logs. The exception is an optimistic conversation conflict, which returns status `400`, type `invalid_request_error`, code `conversation_locked`, and param `conversation`. No part of the stale turn is persisted, so the client can retry the request against the conversation's latest state. This prevents clients from receiving a response ID that cannot be continued after a lock timeout or other database failure without exposing database schema or constraint details.

`AGENTIC_API_SCHEMA_READY` keeps schema changes under supervisor control. Startup performs a read-only compatibility
check and fails if required persistence columns, types, nullability, primary/foreign-key constraints, or the conversation
sequence index are missing, or if the four integer columns still need widening. Apply this upgrade in one transaction
with a DDL-capable migration role before starting the DML-only gateway role. When using `psql`, pass
`-v ON_ERROR_STOP=1` so any statement failure stops the script:

```sql
BEGIN;
ALTER TABLE conversations
    ALTER COLUMN created_at TYPE BIGINT USING created_at::BIGINT;
ALTER TABLE items
    ALTER COLUMN created_at TYPE BIGINT USING created_at::BIGINT,
    ALTER COLUMN seq TYPE BIGINT USING seq::BIGINT;
ALTER TABLE responses
    ALTER COLUMN created_at TYPE BIGINT USING created_at::BIGINT;
DROP INDEX IF EXISTS idx_items_conversation_id;
CREATE UNIQUE INDEX idx_items_conversation_id ON items (conversation_id, seq);
COMMIT;
```

Enable automated backups and point-in-time recovery in the managed database service according to the deployment's recovery-point and recovery-time requirements. Test restores separately; gateway replicas do not create database backups.

## Smoke test

The liveness probe reports whether the gateway process is serving traffic. Startup performs one rollback-only functional
write through the conversation, item, and response tables to verify the configured role and persistence policies.
Readiness then checks the upstream inference service and runs a one-second read-only database, schema, and privilege
probe, avoiding write amplification from orchestrator polling.

```console
curl --fail http://127.0.0.1:9000/health
curl --fail http://127.0.0.1:9000/ready
```

The container CI workflow builds the image, verifies that build tools are absent, launches the gateway against a mock upstream, checks both probes, and exercises a stored Responses API request through SQLite persistence. HTTP streaming and WebSockets use the same gateway binary and exposed port; the image does not add a transport proxy.

On `SIGTERM`, the gateway stops accepting connections and gives in-flight requests up to eight seconds to drain before closing the remaining connections. Set an orchestrator termination grace period longer than eight seconds; the default 30-second Kubernetes grace period and the documented 10-second Docker stop timeout both satisfy this requirement.

## Kubernetes and OpenShift security context

The image defaults to UID `10001` and GID `0`. Its working directory is setgid and the entrypoint uses a group-cooperative umask, so new SQLite files remain writable when OpenShift replaces the UID while retaining the group-0 permission model. Do not set a fixed `runAsUser` when the cluster assigns arbitrary UIDs.

A volume mounted at `/var/lib/agentic-api` hides the ownership and mode stored in the image. For SQLite, configure the storage class or pod-level `fsGroup` so the mounted directory is writable by a supplemental group assigned to the container. The example below uses group 0 to match the image; if the cluster assigns a different permitted supplemental group, use that group and ensure the volume root is group-writable and setgid. PostgreSQL deployments do not need this pod-level filesystem setting.

Volumes initialized by an older image may contain SQLite files without group-write permission. Before rotating to an arbitrary UID, repair those volumes once as an administrator with `chmod -R g+rwX /var/lib/agentic-api`.

```yaml
spec:
  securityContext:
    fsGroup: 0
    fsGroupChangePolicy: OnRootMismatch
  containers:
    - name: agentic-api
      securityContext:
        allowPrivilegeEscalation: false
        capabilities:
          drop: ["ALL"]
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
```

Mount writable storage at `/var/lib/agentic-api` only when using SQLite. PostgreSQL deployments do not need a persistent filesystem for the gateway.

For a replicated PostgreSQL deployment, use the
[Kubernetes manifests and operational guide](kubernetes.md).
