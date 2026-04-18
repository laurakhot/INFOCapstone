# Backend Engineer Handoff — Husky Support Slack Bot

## What this bot does

Husky Support is a proactive Slack DM bot that alerts users when their laptop telemetry indicates hardware risk. New telemetry files land in S3, trigger a Lambda, which runs inference via SageMaker and calls the Slack bot to send a personalized DM with direct links to resolution guides (Slack Canvas documents). Users can re-run diagnosis, snooze, or opt out directly from the message.

---

## Architecture overview

```
S3 bucket (new telemetry file)
        │
        │  S3 event trigger
        ▼
   AWS Lambda
        │
        │  POST { "s3_file": "telemetry_2024_01_15.csv" }
        ▼
  SageMaker endpoint
        │
        │  { "problematic_users": [ { username, device, features } ] }
        ▼
   AWS Lambda (processes SageMaker response)
        │
        │  calls handle_lambda_payload(payload)
        ▼
  alerts/sender.py  ← PRIMARY INTEGRATION POINT
        │
        │  resolves Slack user ID, opens DM, sends Block Kit message
        ▼
     User's Slack DM
        │
        ├─► Canvas guide links unfurl inline as cards
        ├─► [Re-run diagnosis]  [Snooze for an hour]  [Don't remind me again]
        └─► User clicks "Re-run diagnosis" → backend_client.run_diagnosis() [STUBBED]
```

The Slack bot runs in **Socket Mode** — no public URL, no inbound HTTP. It connects to Slack over a persistent WebSocket started by `python app.py`.

---

## End-to-end data schemas

### 1. S3 trigger → Lambda input

```json
{ "s3_file": "telemetry_2024_01_15.csv" }
```

Lambda reads this file from S3 and sends it to SageMaker. Only the current file is processed — no reprocessing of prior files.

---

### 2. Lambda → SageMaker request

```json
{ "s3_file": "telemetry_2024_01_15.csv" }
```

---

### 3. SageMaker → Lambda response

SageMaker returns one record per user whose telemetry crossed a risk threshold. Healthy users are omitted.

```json
{
  "problematic_users": [
    {
      "username": "jdoe",
      "device": "HP EliteBook 840",
      "features": ["avg_memory_utilization", "uptime_days"]
    },
    {
      "username": "asmith",
      "device": "HP EliteBook 8",
      "features": ["avg_battery_health"]
    }
  ]
}
```

**Field definitions:**

| Field | Type | Description |
|-------|------|-------------|
| `username` | string | Slack work alias — must match the user's Slack `profile.name` or `profile.display_name` |
| `device` | string | Device display name shown in the alert message. Must contain `"hp"`, `"windows"`, `"elitebook"`, or `"thinkpad"` (case-insensitive) for Windows article routing |
| `features` | list[string] | Flagged metric keys. Every key in this list is surfaced in the alert — no further threshold filtering happens in the bot |

---

### 4. Lambda → Slack bot call

Lambda passes the **full SageMaker response payload** directly to `handle_lambda_payload()`:

```python
from alerts.sender import handle_lambda_payload

payload = {
    "problematic_users": [
        {"username": "jdoe", "device": "HP EliteBook 840", "features": ["avg_memory_utilization", "uptime_days"]},
        {"username": "asmith", "device": "HP EliteBook 8", "features": ["avg_battery_health"]}
    ]
}

handle_lambda_payload(payload)
```

`handle_lambda_payload()` iterates the list and calls `send_alert()` once per user. Failures for individual users are caught and logged — one user's failure does not stop the rest.

**Alternatively**, Lambda can call `send_alert()` per user directly:

```python
from alerts.sender import send_alert

send_alert(
    username="jdoe",
    device="HP EliteBook 840",
    features=["avg_memory_utilization", "uptime_days"]
)
```

---

## `alerts/sender.py` — deep dive

### Public API

| Function | Signature | Purpose |
|----------|-----------|---------|
| `handle_lambda_payload` | `(payload: dict) -> None` | **Preferred entry point.** Processes full SageMaker JSON, calls `send_alert()` per user |
| `send_alert` | `(username: str, device: str, features: list) -> None` | Sends a single user's alert DM end-to-end |

### `send_alert()` execution flow

```
1. _load_opt_outs()
      └─ reads data/opt_outs.json
      └─ if username present → return early, no alert sent

2. WebClient(token=SLACK_BOT_TOKEN)
      └─ authenticates all Slack API calls

3. _get_user(client, username)
      └─ paginates users.list (200/page)
      └─ matches on profile.name or profile.display_name
      └─ returns (user_id, first_name) — raises ValueError if not found

4. conversations.open(users=[user_id])
      └─ opens (or retrieves) the bot DM channel
      └─ returns channel_id

5. has_replacement = any(is_replacement(k) for k in features)
      └─ checks rag/escalation_logic.md — True if any feature is type=replacement

6. canvas_entries = [(get_canvas_url(k), METRIC_BUTTON_LABELS[k]) for k in features]
      └─ looks up the Slack Canvas URL for each flagged metric
      └─ returns None for canvas_url if not yet configured in escalation_logic.md

7. build_alert_blocks(first_name, device, canvas_entries, has_replacement)
      └─ builds Slack Block Kit blocks (see message structure below)
      └─ returns [] if canvas_entries is empty → return early

8. chat.postMessage(channel_id, text=fallback_text, blocks=blocks, unfurl_links=True)
      └─ sends the DM
      └─ unfurl_links=True causes Slack to expand Canvas URLs as inline cards
```

---

## Slack message structure

The alert DM is a single Block Kit section block + one actions block:

```
┌──────────────────────────────────────────────────────────────┐
│  Hey {first_name} 👋                                         │
│                                                              │
│  Our system detected some risk factors that may cause your   │
│  laptop to crash. 😵 Here are guides based on what we found  │
│  on your *{device}*:                                         │
│                                                              │
│  • Reduce Memory Usage          ← mrkdwn link → Canvas URL  │
│  • Restart your Laptop          ← mrkdwn link → Canvas URL  │
│                                                              │
│  After you resolved these risk factors, click               │
│  "re-run diagnosis" below.                                   │
├──────────────────────────────────────────────────────────────┤
│  [Re-run diagnosis]  [Snooze for an hour]  [Don't remind me] │
└──────────────────────────────────────────────────────────────┘

Below the blocks, Slack auto-unfurls Canvas links as cards:
┌──────────────────────────────────┐
│ 📋 Reduce Memory Usage  · Canvas │
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ 📋 Sleep-Deprived Laptop · Canvas│
└──────────────────────────────────┘
```

**Key design decisions:**
- No confidence scores are shown — every feature in `features[]` is presented equally
- Canvas URLs are embedded as `<url|label>` mrkdwn links so they render as blue clickable text AND auto-unfurl as Canvas cards
- The `fallback_text` parameter (plain text) is used for push notifications and screen readers
- `unfurl_links=True` is explicitly set on `chat.postMessage` to guarantee unfurling

---

## `rag/escalation_logic.md` — routing table

Controls per-metric behavior. Editable by the data team without code changes.

```markdown
| metric_key             | type         | canvas_url                          |
|------------------------|--------------|-------------------------------------|
| avg_processor_time     | replacement  | https://...slack.com/docs/T.../F... |
| max_cpu_usage          | self-service | https://...slack.com/docs/T.../F... |
| avg_memory_utilization | self-service | https://...slack.com/docs/T.../F... |
| avg_battery_health     | replacement  | https://...slack.com/docs/T.../F... |
| uptime_days            | self-service | https://...slack.com/docs/T.../F... |
| p90_cpu_temp           | self-service | https://...slack.com/docs/T.../F... |
```

**Column reference:**

| Column | Values | Effect |
|--------|--------|--------|
| `type` | `self-service` | Canvas guides shown; "Re-run diagnosis" button available |
| `type` | `replacement` | Same UX — Canvas guides shown; sets `has_replacement=True` flag for message copy variation |
| `canvas_url` | Slack Canvas URL | Embedded as clickable link in DM; unfurls as Canvas card. Use `-` if not yet available |

**Parsed by** `rag/articles.py → _load_escalation_table()`. The table is loaded once at module import time. Restart the bot after editing this file.

---

## Supported metrics (current feature set)

| metric_key | Display label | Button label | Type | Canvas configured |
|---|---|---|---|---|
| `avg_processor_time` | Processor Time | Check Processor Performance | replacement | ✅ |
| `max_cpu_usage` | Max CPU Usage | Optimize CPU Usage | self-service | ✅ |
| `avg_memory_utilization` | Average Memory Utilization | Reduce Memory Usage | self-service | ✅ |
| `avg_battery_health` | Battery Health | Check Battery Health | replacement | ✅ |
| `uptime_days` | System Uptime | Restart your Laptop | self-service | ✅ |
| `p90_cpu_temp` | CPU Temperature | Cool Down your Laptop | self-service | ✅ |

**Do not include in `features[]`:** any metric key not in this table will be shown with a plain text fallback label and no Canvas link.

---

## `rag/articles.py` — key functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_canvas_url` | `(metric_key: str) -> Optional[str]` | Returns the Canvas URL for a metric from escalation_logic.md; `None` if not configured |
| `is_replacement` | `(metric_key: str) -> bool` | Returns `True` if metric type is `replacement` |
| `get_article` | `(metric_key: str, device: str) -> str` | Returns a KB article as Slack mrkdwn for Q&A responses. Platform inferred from device name |
| `_detect_platform` | `(device: str) -> str` | Returns `"windows"` if device contains `hp`, `windows`, `elitebook`, or `thinkpad`; else `"mac"` |

Article files live at `rag/articles/{platform}_{metric_key}.md`. These are used for the reactive Q&A flow (user DMs the bot), not the proactive alert.

---

## Button action handlers (`handlers/actions.py`)

| Action ID | Handler | What it does |
|-----------|---------|--------------|
| `rerun_diagnosis` | `handle_rerun_diagnosis` | Calls `run_diagnosis(username)` [stubbed], posts result in thread, marks original alert resolved |
| `done_selfservice__{metric_key}` | `handle_done_selfservice` | Same as rerun_diagnosis — triggered from article thread reply |
| `done_replacement__{metric_key}` | `handle_done_replacement` | Posts IT handoff confirmation in thread, marks alert resolved |
| `snooze` | `handle_snooze` | Replaces alert message with "⏰ Snoozed" state |
| `opt_out` | `handle_opt_out` | Replaces alert with unsubscribed state; posts optional feedback prompt |
| `opt_out_reason__{reason}` | `handle_opt_out_reason` | Saves reason to `data/opt_outs.json`; replaces feedback prompt with thank-you |

All handlers call `ack()` first (required by Slack Bolt within 3 seconds).

**Re-run diagnosis flow:**
```
User taps "Re-run diagnosis"
  → bot posts "Running a fresh check... ⏳" in thread
  → calls run_diagnosis(username) [currently stubbed → returns {}]
  → updates loading message with "all clear 🎉"
  → strips action buttons from original alert, appends "✅ Resolved" footer
```

---

## Event handlers (`handlers/events.py`)

| Event | Handler | What it does |
|-------|---------|--------------|
| `app_home_opened` | `handle_app_home_opened` | Publishes the App Home tab view via `views.publish` |
| `message` (im only) | `handle_dm_message` | Keyword-based Q&A — matches user text to metric articles |

**Q&A keyword map** (current, in `_build_qa_response()`):

> ⚠️ **Note for backend team:** The keyword map in `handlers/events.py` references some outdated metric keys (`avg_boot_time`, `cpu_count`, `battery_cycle`, `max_memory_pressure`) that have been removed from the feature set. These will fall back to the generic response. Update the keyword map to match the current 6 metrics before launch.

**In-memory alert context** (`_recent_alerts` dict): maps `user_id → {metric_key, device}` for the most recently received alert. Used to provide contextual Q&A if no keyword matches. **Lost on bot restart** — this is intentional for MVP; a persistent store would be needed for production.

---

## `backend_client.py` — re-diagnosis stub

```python
def run_diagnosis(username: str) -> dict:
    # Currently stubbed — always returns {} (all clear)
    # Uncomment real call below when backend exposes GET /diagnose/{username}
```

**To wire up the real backend**, uncomment the `requests.get()` block and set `BACKEND_URL` in `.env`. Expected response shape:

```json
{ "features": ["avg_memory_utilization"] }
```

or `{}` / empty `features` for all clear.

---

## Device name convention

The `device` field from SageMaker drives platform-specific article routing for Q&A:

| Device name contains | Articles served |
|---|---|
| `"hp"`, `"windows"`, `"elitebook"`, `"thinkpad"` (case-insensitive) | `rag/articles/windows_*.md` |
| anything else | `rag/articles/mac_*.md` |

Canvas URLs in the proactive alert are platform-agnostic — the same Canvas link is used regardless of device. Platform routing only matters for the reactive Q&A flow.

---

## Environment variables

| Variable | Required by | Notes |
|---|---|---|
| `SLACK_BOT_TOKEN` | `alerts/sender.py`, all Slack API calls | `xoxb-` token from api.slack.com/apps → OAuth & Permissions |
| `SLACK_APP_TOKEN` | `app.py` (Socket Mode) | `xapp-` token, requires `connections:write` scope |
| `BACKEND_URL` | `backend_client.py` | Base URL for reactive re-diagnosis endpoint — not used in proactive flow |

---

## Running the bot

```bash
# Install dependencies
pip install -r requirements.txt

# Start the bot (persistent WebSocket connection)
python app.py

# Simulate a Lambda call — single user (bot does NOT need to be running for this)
python -m alerts.sender --user=jdoe --device="HP EliteBook 8" \
  --features='["avg_memory_utilization","uptime_days"]'

# Simulate a Lambda call — full payload (from Python)
python -c "
from alerts.sender import handle_lambda_payload
handle_lambda_payload({
  'problematic_users': [
    {'username': 'jdoe', 'device': 'HP EliteBook 8', 'features': ['avg_memory_utilization', 'uptime_days']}
  ]
})
"
```

> **Note:** `send_alert()` and `handle_lambda_payload()` call the Slack API directly and do NOT require `app.py` to be running. `app.py` is only needed for button interactions (re-run diagnosis, snooze, opt-out).

---

## Key files

| File | Role |
|------|------|
| `app.py` | Entry point — starts Socket Mode WebSocket listener, registers all handlers |
| `alerts/sender.py` | **Primary Lambda integration point** — `handle_lambda_payload()` and `send_alert()` |
| `alerts/templates.py` | All Slack Block Kit message builders; defines `METRIC_LABELS` and `METRIC_BUTTON_LABELS` |
| `backend_client.py` | Stub for reactive re-diagnosis — `GET /diagnose/{username}` when backend is ready |
| `handlers/actions.py` | Button click handlers: `rerun_diagnosis`, `snooze`, `opt_out`, opt-out feedback |
| `handlers/events.py` | DM keyword Q&A handler + App Home tab publisher |
| `rag/articles.py` | `get_canvas_url()`, `is_replacement()`, `get_article()` — reads escalation_logic.md |
| `rag/escalation_logic.md` | **Data-team editable** routing table: metric → type + Canvas URL |
| `rag/articles/{platform}_{metric}.md` | KB article content for Q&A responses (one per platform per metric) |
| `data/opt_outs.json` | Persistent opt-out store — auto-created on first opt-out |

---

## Required Slack OAuth scopes

App: **A0APD0GGK5G** → api.slack.com/apps → OAuth & Permissions → Bot Token Scopes

| Scope | Used for |
|-------|---------|
| `chat:write` | Send DMs and thread replies |
| `im:write` | Open DM channels (`conversations.open`) |
| `users:read` | Resolve username → user ID (`users.list`) |
| `conversations:history` | Read original alert message to strip action buttons on resolve |

---

## Open questions and gaps before deployment

| # | Question | Impact | Owner |
|---|---|---|---|
| 1 | **SageMaker `features` field name** — the bot expects the key `"features"` in the per-user object. Confirm SageMaker returns this exact key (not `"flagged_metrics"` or `"metrics"`). | Lambda must rename if different | ML / Backend |
| 2 | **Healthy users in SageMaker response** — are they returned under a separate key or omitted? Lambda should only pass `problematic_users` to the bot. | Lambda filtering logic | ML |
| 3 | **Device name from telemetry** — confirm the `device` field in SageMaker output contains a hostname with `"hp"`, `"elitebook"`, or `"windows"` so Windows articles are routed correctly. | Q&A article routing | Backend / Data |
| 4 | **Alert deduplication** — if Lambda is triggered multiple times for the same user in one day, the user receives duplicate DMs. Deduplication logic needs to live in Lambda or a shared store. | User experience | Backend |
| 5 | **Opt-out persistence** — `data/opt_outs.json` is a local file. If the bot runs on multiple instances or restarts, opt-outs from one instance are invisible to others. Needs DynamoDB or similar before production. | Reliability | Backend |
| 6 | **Lambda → bot invocation method** — `send_alert()` is currently called by importing the module (assumes co-located code). If the Slack bot runs as a separate service, Lambda needs an HTTP endpoint or SQS queue. Architecture not yet decided. | Deployment topology | Backend |
| 7 | **`run_diagnosis()` real endpoint** — `backend_client.py` is stubbed. When the backend exposes `GET /diagnose/{username}`, uncomment the `requests.get()` block and set `BACKEND_URL`. The bot expects a `features` list in the response. | Re-run diagnosis UX | Backend |
| 8 | **Q&A keyword map outdated** — `handlers/events.py` references old metric keys (`avg_boot_time`, `cpu_count`, `battery_cycle`). Update `keyword_map` dict to match current 6 metrics before launch. | Q&A accuracy | Frontend |
| 9 | **S3 bucket name and IAM permissions** — Lambda needs `s3:GetObject` on the telemetry bucket and `sagemaker:InvokeEndpoint` on the model endpoint. | Infrastructure | DevOps |
