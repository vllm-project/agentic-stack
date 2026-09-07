# Run Agentic API on Kubernetes with kind

This guide runs a single vLLM Agentic API replica on a local
[kind](https://kind.sigs.k8s.io/) cluster. It is intended for development and smoke testing, not production deployment.

The example assumes that vLLM or an llm-d inference gateway is already running on the
host. Agentic API runs in kind and reaches the upstream through `host.docker.internal`.

!!! note

    The example stores response state in PostgreSQL backed by a persistent volume,
    so conversations survive pod restarts. The single-replica PostgreSQL Deployment
    below is sized for local development; use a managed or highly available
    PostgreSQL service for production. For a throwaway smoke test without
    persistence, set `DATABASE_URL` to `sqlite:///tmp/agentic_api.db` instead and
    skip the PostgreSQL section.

## Use llm-d as the inference backend

The same topology can place [llm-d](https://llm-d.ai/) between Agentic API and
the vLLM serving stack. This is the deployment pattern validated on GPU hardware:
llm-d owns inference routing and presents an OpenAI-compatible upstream endpoint,
while Agentic API owns response state, tool orchestration, and the agent loop.

```mermaid
flowchart LR
    C["Codex or Claude Code"]
    A["Agentic API\n(Kubernetes / kind)"]
    L["llm-d\n(router / inference gateway)"]
    VC["vLLM CPU server\n(rendering / scheduling)"]
    VG["vLLM GPU worker\n(token generation)"]
    T["Agentic API tools\n(web search / MCP / client tools)"]

    C -->|"Responses or Messages"| A
    A -->|"model request"| L
    L --> VC
    VC --> VG
    VG -->|"generated tokens"| VC
    VC -->|"streamed model events"| L
    L -->|"streamed response"| A
    A -->|"tool call"| T
    T -->|"tool result for next model turn"| A
    A -->|"final response"| C

    classDef client fill:#1a3a5c,color:#e0e0e0
    classDef gateway fill:#1a5c2a,color:#e0e0e0
    classDef router fill:#5c3a1a,color:#e0e0e0
    classDef worker fill:#4b3a6b,color:#e0e0e0
    class C client
    class A,T gateway
    class L router
    class VC,VG worker
```

For this arrangement, configure Agentic API's upstream URL to the llm-d
endpoint rather than directly to a vLLM server. The Deployment below uses a
host alias because the local kind pod reaches the tested llm-d/vLLM stack on the
host; in a Kubernetes deployment where llm-d is a Service, use the Service DNS
name instead. The tool loop remains in Agentic API: a model tool call is
executed or returned to the client, and the resulting turn is sent back through
llm-d for inference.

## Prerequisites

Install and verify:

```console
docker version
kind version
kubectl version --client
```

You need an inference endpoint reachable from Docker at
`host.docker.internal:5050`. This can be vLLM directly, or the llm-d endpoint
fronting the vLLM workers. For a direct vLLM smoke test, start vLLM on the host with:

```console
vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --served-model-name Qwen/Qwen3-30B-A3B-FP8 qwen3-30b-a3b-fp8 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 5050
```

The second `--served-model-name` entry publishes a slash-free alias alongside the canonical name. Direct API calls
work with either, but `agentic run claude` (0.4.0+) rejects model names containing `/`, so testing the deployment
with the [harness CLI](../guides/harness-cli-testing.md) needs the alias.

The `--host 0.0.0.0` setting matters because the vLLM process must accept traffic
from the Docker network.

## Build the local image

Use the repository's `Dockerfile`, the same multi-stage, digest-pinned build that CI
publishes. It builds on macOS (including Apple Silicon) as well as Linux, and the
`.dockerignore` file keeps local build output out of the Docker context.

Build the image and load it into kind:

```console
docker build -t agentic-api:kind .
kind create cluster --name agentic-api
kind load docker-image agentic-api:kind --name agentic-api
```

If the cluster already exists, skip `kind create cluster` and load the image again
after rebuilding it.

### Podman

Podman can run kind through its experimental provider. Start a Podman machine and
build the image with Podman first - Podman stores unqualified image names under the
`localhost/` prefix, so use that name consistently when loading the image and in the
Deployment's `image:` field:

```console
podman machine start
podman build -t agentic-api:kind .
KIND_EXPERIMENTAL_PROVIDER=podman kind create cluster --name agentic-api-podman
KIND_EXPERIMENTAL_PROVIDER=podman kind load docker-image localhost/agentic-api:kind --name agentic-api-podman
```

In the Deployment below, set `image: localhost/agentic-api:kind`, and replace
`host.docker.internal` with `host.containers.internal`. The latter is the hostname
Podman provides for reaching services on the host.

## Deploy PostgreSQL

Response state lives in PostgreSQL. Create a Secret holding both the database
password and the full connection URL that Agentic API consumes:

```console
PGPASS=$(openssl rand -hex 16)
kubectl create secret generic agentic-api-postgres \
  --from-literal=password="$PGPASS" \
  --from-literal=database-url="postgres://postgres:${PGPASS}@agentic-api-postgres.default.svc.cluster.local:5432/agentic_api"
```

Save the following as `postgres-kind.yaml` and apply it. The PersistentVolumeClaim
uses kind's default `standard` StorageClass, so the data survives pod restarts:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agentic-api-postgres
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-api-postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentic-api-postgres
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: agentic-api-postgres
    spec:
      containers:
        - name: postgres
          image: postgres:17-alpine
          env:
            - name: POSTGRES_DB
              value: agentic_api
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: agentic-api-postgres
                  key: password
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          ports:
            - name: postgres
              containerPort: 5432
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres", "-d", "agentic_api"]
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: agentic-api-postgres
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-api-postgres
spec:
  selector:
    app: agentic-api-postgres
  ports:
    - name: postgres
      port: 5432
      targetPort: postgres
```

```console
kubectl apply -f postgres-kind.yaml
kubectl rollout status deployment/agentic-api-postgres
```

Agentic API runs its schema migrations automatically on startup, so no manual
database initialization is needed.

## Deploy Agentic API

Apply the following Deployment and Service. The `host.docker.internal` address is
available in Docker Desktop. On native Linux Docker, that hostname does not resolve
inside the cluster, so you must also add the `hostAliases` block shown in the section
below before applying the manifest.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentic-api
  template:
    metadata:
      labels:
        app: agentic-api
    spec:
      containers:
        - name: agentic-api
          image: agentic-api:kind
          imagePullPolicy: IfNotPresent
          args:
            - --llm-api-base
            - http://host.docker.internal:5050
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: agentic-api-postgres
                  key: database-url
          ports:
            - name: http
              containerPort: 9000
          startupProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 5
            failureThreshold: 60
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            periodSeconds: 5
            failureThreshold: 30
          livenessProbe:
            httpGet:
              path: /health
              port: http
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-api
spec:
  selector:
    app: agentic-api
  ports:
    - name: http
      port: 9000
      targetPort: http
```

The `startupProbe` matters: the server does not bind its HTTP listener until the
configured LLM endpoint reports healthy, so while a large model is still loading,
`/health` is unreachable. Without the startup probe, the liveness probe would kill
the container after about 30 seconds and leave the pod in `CrashLoopBackOff`. The
settings above allow up to five minutes; raise `failureThreshold` for models with
longer load times.

Save the YAML as `agentic-api-kind.yaml`, then apply it:

```console
kubectl apply -f agentic-api-kind.yaml
kubectl rollout status deployment/agentic-api
kubectl get pods,svc
```

On native Linux Docker, `host.docker.internal` never resolves inside the pod, so
add this block below `spec.template.spec` in the Deployment before applying the
manifest:

```yaml
      hostAliases:
        - ip: "172.18.0.1"
          hostnames:
            - host.docker.internal
```

The IP must be the gateway of the `kind` Docker network, not the default bridge:
kind nodes run on their own network, so the usual `172.17.0.1` bridge gateway is
not reachable from the pod. Confirm the address with:

```console
docker network inspect kind \
  --format '{{range .IPAM.Config}}{{.Gateway}} {{end}}'
```

## Call the API

Forward the Service to the host:

```console
kubectl port-forward service/agentic-api 9000:9000
```

In another terminal, check both probes:

```console
curl http://localhost:9000/health
curl http://localhost:9000/ready
```

Make a stateful Responses API request:

```console
curl http://localhost:9000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-FP8",
    "input": "Say hello from kind."
  }'
```

The response includes an `id`. Continue the conversation by passing it as
`previous_response_id` - this is the stateful part, served from the response
store rather than client-supplied history:

```console
curl http://localhost:9000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-FP8",
    "input": "Repeat what you just said.",
    "previous_response_id": "<id from the previous response>"
  }'
```

The model name must match the model served by vLLM. View the server logs while
testing:

```console
kubectl logs -f deployment/agentic-api
```

### Verify persistence

Because response state is in PostgreSQL rather than inside the pod, conversations
survive a pod restart. Verify it by deleting the pod and continuing the same
conversation afterwards:

```console
kubectl delete pod -l app=agentic-api
kubectl rollout status deployment/agentic-api
kubectl port-forward service/agentic-api 9000:9000
```

Re-run the continuation request with the same `previous_response_id`; the model
still sees the earlier turns. The same holds if the PostgreSQL pod restarts, since
its data directory lives on the PersistentVolumeClaim.

## Optional web search

To enable the gateway-executed `web_search` built-in tool, add the provider settings
to the Deployment’s container environment:

```yaml
            - name: YOU_API_KEY
              valueFrom:
                secretKeyRef:
                  name: agentic-api-secrets
                  key: you-api-key
            - name: YOU_API_BASE_URL
              value: https://ydc-index.io
            # Opt Claude Code's WebSearch function into gateway execution.
            - name: MESSAGES_GATEWAY_TOOL_ALIASES
              value: WebSearch=web_search
```

Create the secret before applying the Deployment:

```console
kubectl create secret generic agentic-api-secrets \
  --from-literal=you-api-key="$YOU_API_KEY"
```

Do not commit API keys to the manifest or source tree.

## Optional: deploy with llm-d

`--llm-api-base` accepts any OpenAI-compatible endpoint, not only a single vLLM
server. Pointing it at an inference gateway backed by
[llm-d](https://llm-d.ai/) and the
[Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
lets one Agentic API instance serve multiple models (the gateway routes on the
request's `model` field) and lets the endpoint picker (EPP) place each request
on the vLLM replica that already holds the matching KV-cache prefix. Because
stateful Responses continuations rehydrate the full conversation history,
prefix-aware placement substantially reduces continuation latency for
multi-turn workloads; see the measurements in
[ADR-04](https://github.com/vllm-project/agentic-api/issues/69).

The steps below add the routing plane to the kind cluster from this guide.
They keep vLLM on the host, as elsewhere in this guide; each host server is
represented inside the cluster by a small TCP proxy pod so the `InferencePool`
can select and monitor it. On a real cluster, vLLM runs as in-cluster pods and
the pool selects them directly - the proxy pods are only the local-development
bridge, and the EPP scrapes vLLM metrics through them transparently.

### Install the routing plane

Install the Gateway API and Inference Extension CRDs, then
[Agentgateway](https://agentgateway.dev/) as the gateway implementation:

```console
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.0/standard-install.yaml
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.5.0/manifests.yaml
helm upgrade -i --create-namespace --namespace agentgateway-system --version v1.0.0 \
  agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds
helm upgrade -i --namespace agentgateway-system --version v1.0.0 \
  agentgateway oci://cr.agentgateway.dev/charts/agentgateway \
  --set inferenceExtension.enabled=true
```

### Create the pool backends and Gateway

Save and apply the following as `llm-pool.yaml`. The proxy pod forwards pool
traffic to the host vLLM server; the IP is the `kind` network gateway described
in the `hostAliases` section above. To scale the pool, start more vLLM servers
on additional host ports and add one labeled proxy pod per server:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-replica-1
  labels:
    app: vllm-pool
spec:
  containers:
    - name: proxy
      image: alpine/socat:1.8.0.0
      args: ["TCP-LISTEN:8000,fork,reuseaddr", "TCP:172.18.0.1:5050"]
      ports:
        - containerPort: 8000
      readinessProbe:
        tcpSocket:
          port: 8000
        periodSeconds: 5
---
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: agentgateway
  listeners:
    - name: http
      port: 80
      protocol: HTTP
```

On macOS or Windows (Docker Desktop), the `172.18.0.1` kind network gateway is not
reachable from pods; use `TCP:host.docker.internal:5050` as the socat target
instead. Note that the pod's TCP readiness probe only checks socat's listener, not
the forwarding target, so a wrong address here shows up as hanging requests rather
than an unready pod - verify the path with the `/v1/models` check below before
wiring anything else to it.

### Install the InferencePool and EPP

The Helm chart installs the `InferencePool`, the EPP, and an `HTTPRoute` bound
to the Gateway:

```console
helm install vllm-pool \
  --dependency-update \
  --set inferencePool.modelServers.matchLabels.app=vllm-pool \
  --set provider.name=none \
  --set experimentalHttpRoute.enabled=true \
  --version v1.5.0 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
```

On arm64 hosts the chart's default EPP image is amd64-only and crashes with
`exec format error`; switch it to the multi-arch llm-d scheduler build, which
is the same EPP framework:

```console
kubectl set image deployment/vllm-pool-epp \
  epp=ghcr.io/llm-d/llm-d-inference-scheduler:latest
```

Verify the path end to end before involving Agentic API:

```console
kubectl port-forward svc/inference-gateway 8080:80 &
curl http://localhost:8080/v1/models
```

### Point Agentic API at the gateway

In the Deployment from this guide, replace the `--llm-api-base` value with the
gateway Service URL and apply it again:

```yaml
          args:
            - --llm-api-base
            - http://inference-gateway.default.svc.cluster.local:80
```

The readiness probe works unchanged: `/ready` checks vLLM `/health` through the
gateway, which routes it to a pool member. Requests to `/v1/responses` now flow
client -> Agentic API -> gateway -> EPP-selected vLLM replica.

## Troubleshooting

### The pod stays unready

`/ready` checks the configured vLLM `/health` endpoint. Inspect the pod logs and verify
that vLLM is listening on `0.0.0.0:5050` and that the configured host address is
reachable from Docker:

```console
kubectl describe pod -l app=agentic-api
kubectl logs deployment/agentic-api
docker run --rm curlimages/curl:8.10.1 \
  http://host.docker.internal:5050/health
```

On native Linux Docker, run the connectivity check on the `kind` network against
its gateway instead, since `host.docker.internal` does not resolve:

```console
docker run --rm --network kind curlimages/curl:8.10.1 \
  http://172.18.0.1:5050/health
```

### The image is not refreshed

kind uses the image already loaded into its node. Rebuild and load it explicitly, then
restart the Deployment:

```console
docker build -t agentic-api:kind .
kind load docker-image agentic-api:kind --name agentic-api
kubectl rollout restart deployment/agentic-api
```

### Inspect the rendered configuration

```console
kubectl get deployment agentic-api -o yaml
kubectl get events --sort-by=.lastTimestamp
```

## Clean up

Delete the Kubernetes resources and the kind cluster when finished. If you deployed
the optional llm-d routing plane, remove it first:

```console
helm uninstall vllm-pool
kubectl delete -f llm-pool.yaml
helm uninstall agentgateway --namespace agentgateway-system
helm uninstall agentgateway-crds --namespace agentgateway-system
```

Then delete the Agentic API and PostgreSQL resources and the cluster. Deleting
the PersistentVolumeClaim removes the stored response state:

```console
kubectl delete -f agentic-api-kind.yaml
kubectl delete -f postgres-kind.yaml
kubectl delete secret agentic-api-postgres
kind delete cluster --name agentic-api
```

If you used the Podman provider, delete that cluster as well:

```console
KIND_EXPERIMENTAL_PROVIDER=podman kind delete cluster --name agentic-api-podman
```

The temporary `agentic-api-kind.yaml`, `postgres-kind.yaml`, and `llm-pool.yaml`
files can then be removed from the repository checkout.
