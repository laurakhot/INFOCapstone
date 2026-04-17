# Backend Engineer Handoff — Husky Support Slack Bot

## What this bot does

Husky Support is a proactive Slack DM bot that alerts Windows users when their laptop telemetry crosses BSOD-risk thresholds. The proactive flow is fully automated: new telemetry files land in S3, trigger a Lambda, which runs inference via SageMaker and calls the Slack bot to notify affected users. Reactive (user-initiated) re-diagnosis is a secondary use case and currently stubbed.

---

## Architecture overview

```
S3 bucket (new telemetry file)
        │
        │  S3 event trigger (file name as payload)
        ▼
   AWS Lambda
        │
        │  POST to SageMaker endpoint
        │  { "s3_file": "telemetry_2024_01_15.csv" }
        ▼
  SageMaker endpoint
        │
        │  JSON — one record per problematic user
        │  { "problematic_users": [ { "username", "device", "flagged_metrics" } ] }
        ▼
   AWS Lambda (processes response)
        │
        │  for each problematic user:
        │  send_alert(username, device, predictions)
        ▼
  alerts/sender.py  (this repo)
        │
        │  Slack DM with flagged metrics + canvas link
        ▼
     User's Slack DM
        │
        └─► "📋 Open Resolution Guide" → pre-built Slack Canvas
```

The Slack bot runs in **Socket Mode** (no public URL needed). It connects to Slack over a persistent WebSocket — no inbound HTTP endpoints to expose or proxy.

**Priority:** Proactive flow (S3 → Lambda → SageMaker → Slack) is the primary use case. Reactive re-diagnosis (user clicks "Done, re-run") is secondary and currently stubbed.

---

## End-to-end data schemas

### 1. Input to Lambda — S3 trigger payload

```json
{
  "s3_file": "telemetry_2024_01_15.csv"
}
```

Lambda receives the file name (not the full path) as the variable to process. Only this file is evaluated — no reprocessing of prior files. No state management required.

---

### 2. SageMaker endpoint — request

Lambda sends the S3 file reference to SageMaker:

```json
{
  "s3_file": "telemetry_2024_01_15.csv"
}
```

---

### 3. SageMaker endpoint — response

SageMaker returns the **full row** for every user with at least one problematic feature. Healthy users are omitted (or grouped separately — see open questions).

```json
{
  "problematic_users": [
    {
      "username": "jdoe",
      "device": "HP-ELITEBOOK-001",
      "flagged_metrics": ["max_cpu_usage", "uptime_days"]
    },
    {
      "username": "asmith",
      "device": "HP-840-G9",
      "flagged_metrics": ["p90_boot_time"]
    }
  ]
}
```

> **Note:** `cpu_count` and `avg_memory_utilization` have been **removed as model features** and will not appear in `flagged_metrics`.

---

### 4. Slack bot call — per user

Lambda iterates `problematic_users` and calls `send_alert()` once per user:

```python
from alerts.sender import send_alert

send_alert(
    username="jdoe",               # from SageMaker response
    device="HP-ELITEBOOK-001",     # from SageMaker response — must contain "hp" or "windows" for Windows article routing
    predictions={
        "max_cpu_usage": 1,        # convert flagged_metrics list → dict of {key: 1}
        "uptime_days": 1,
    }
)
```

**Lambda's job** is to translate `flagged_metrics: ["max_cpu_usage", "uptime_days"]` into `predictions: {"max_cpu_usage": 1, "uptime_days": 1}`. Every key in `predictions` is treated as a confirmed threshold breach — no further filtering happens inside the bot.

---

## `alerts/sender.py` — deep dive

### Functions defined

| Function | Purpose |
|---|---|
| `send_alert(username, device, predictions)` | Public entry point — orchestrates the full alert flow |
| `_get_user(client, username)` | Resolves a Slack username to `(user_id, first_name)` |
| `_load_opt_outs()` | Reads `data/opt_outs.json`; returns `{}` if file missing |

---

### Execution flow (step by step)

```
1. _load_opt_outs()
      └─ reads data/opt_outs.json
      └─ if username is present → return early, no alert sent

2. WebClient(token=SLACK_BOT_TOKEN)
      └─ authenticates all subsequent API calls

3. _get_user(client, username)
      └─ calls users.list (paginated, 200 users/page)
      └─ matches on profile.name or profile.display_name
      └─ extracts first_name from profile.first_name or real_name.split()[0]
      └─ returns (user_id, first_name)   ← raises ValueError if not found

4. conversations.open(users=[user_id])
      └─ opens (or retrieves) the bot's DM channel with the user
      └─ returns channel_id

5. has_replacement = any(is_replacement(k) for k in predictions)
      └─ checks rag/escalation_logic.md — True if any metric requires hardware replacement

6. build_alert_blocks(first_name, device, predictions, has_replacement)
      └─ returns Slack Block Kit blocks (list of dicts)
      └─ if predictions is empty → returns [] → return early, no message sent

7. chat.postMessage(channel=channel_id, text=fallback_text, blocks=blocks)
      └─ sends the DM
```

---

### Slack message structure

```
┌─────────────────────────────────────────────┐
│ Hey {first_name} 👋 Our system detected...  │  ← intro text
├─────────────────────────────────────────────┤
│ Here's what we found on your *{device}*:    │
├─────────────────────────────────────────────┤
│ ⚠️ Max CPU Usage                            │  ← one row per flagged metric
│ ⚠️ System Uptime                            │
├─────────────────────────────────────────────┤
│ [Optimize CPU] [Restart my Laptop]          │  ← one button per metric
│ [Remind me later] [Don't remind me again]   │  ← management buttons
└─────────────────────────────────────────────┘
```

Replacement flow (when `has_replacement=True`) replaces per-metric buttons with `[Backup my Data]` + `[Request Laptop Replacement]`.

---

### Authentication / user identification

| What | How |
|---|---|
| **API auth** | `SLACK_BOT_TOKEN` (`xoxb-`) passed to `WebClient` — authorizes every Slack API call |
| **User identification** | `username` string → `users.list` scan → `user_id` (e.g. `U0XXXXXXX`) |

The `user_id` is used to open the DM channel via `conversations.open`. Slack does not expose a "find user by username" endpoint, so the bot paginates all workspace members and matches on `profile.name` or `profile.display_name`. This costs one API call per 200 users. For large workspaces, Lambda can pre-resolve and cache user IDs to reduce latency.

---

## Supported metrics (post-meeting)

`cpu_count` and `avg_memory_utilization` have been removed from the model feature set. Do not include them in `flagged_metrics`.

| metric_key | Display label | Type | Article |
|---|---|---|---|
| `max_cpu_usage` | Max CPU Usage | self-service | `windows_max_cpu_usage.md` |
| `uptime_days` | System Uptime | self-service | `windows_uptime_days.md` |
| `avg_boot_time` | Boot Time | self-service | `windows_avg_boot_time.md` ⚠️ |
| `p90_boot_time` | Worst Boot Time | self-service | `windows_p90_boot_time.md` ⚠️ |
| `memory_size_gb` | Memory Capacity | **replacement** | `not-self-serviceable.md` |

⚠️ = Windows article file does not yet exist — bot will show fallback text for these metrics until the article is created.

Article routing is controlled by `rag/escalation_logic.md` — editable by the data team without code changes.

---

## Device name convention

The device name in the SageMaker response drives which article variant is shown:

| Device name contains | Article folder |
|---|---|
| `"hp"` or `"windows"` | `rag/articles/windows_*.md` |
| anything else | `rag/articles/mac_*.md` |

Since we are prioritizing Windows (BSOD), ensure SageMaker returns device names that include `"hp"` or `"windows"` (case-insensitive). The hostname from the telemetry row is the right value to use here.

---

## Environment variables

| Variable | Required by | Notes |
|---|---|---|
| `SLACK_BOT_TOKEN` | Everything | `xoxb-` token from api.slack.com/apps → OAuth & Permissions |
| `SLACK_APP_TOKEN` | `app.py` (Socket Mode) | `xapp-` token, requires `connections:write` scope |
| `BACKEND_URL` | `backend_client.py` | Base URL for reactive re-diagnosis — not used in proactive flow |
| `SLACK_CHANNEL_ID` | `demo_canvas.py` only | Not used by the main bot |

---

## Running the bot

```bash
# Install dependencies
pip install -r requirements.txt

# Start the bot (keeps running, connects via WebSocket)
python app.py

# Simulate a Lambda call for testing (bot must be running in another terminal)
python alerts/sender.py \
  --user=jdoe \
  --device="HP-ELITEBOOK-001" \
  --predictions='{"max_cpu_usage":1,"uptime_days":1}'

# Demo canvas link post (standalone, no bot needed)
python demo_canvas.py
```

---

## Key files

| File | What it does |
|---|---|
| `app.py` | Entry point — starts Socket Mode listener |
| `alerts/sender.py` | **Primary integration point** — Lambda calls `send_alert()` per problematic user |
| `backend_client.py` | Secondary integration point — stub for reactive re-diagnosis (`GET /diagnose/{username}`) |
| `alerts/templates.py` | All Slack Block Kit message builders |
| `handlers/actions.py` | Button click handlers (Done, Snooze, Opt-out) |
| `handlers/events.py` | DM message handler + App Home |
| `rag/articles.py` | Article loader — platform-aware routing |
| `rag/escalation_logic.md` | Routing table — data team editable |
| `rag/articles/windows_*.md` | Windows KB articles (one per metric) |
| `data/opt_outs.json` | Persistent opt-out store (JSON file, auto-created) |
| `demo_canvas.py` | Standalone demo — posts pre-built canvas link to a channel |

---

## Required Slack OAuth scopes

Scopes configured at api.slack.com/apps → **A0APD0GGK5G** → OAuth & Permissions → Bot Token Scopes.

| Scope | Used for |
|---|---|
| `chat:write` | Send DMs and thread replies |
| `im:write` | Open DM channels (`conversations.open`) |
| `users:read` | Resolve username → user ID |
| `conversations:history` | Read original alert message when updating it |
| `canvases:write` | Canvas creation (paid plan only — not currently used) |
| `canvases:read` | Canvas metadata (paid plan only — not currently used) |

---

## Open questions and gaps before deployment

These need resolution before the proactive flow goes live.

| # | Question | Impact | Owner |
|---|---|---|---|
| 1 | **SageMaker response schema** — does it return `flagged_metrics` as a flat list of strings, or as `{metric_key: value}` pairs? The bot expects a dict. Lambda must translate. | Lambda transform logic | Backend / ML |
| 2 | **Healthy users in SageMaker response** — are they returned under a `healthy_users` key or simply omitted? Lambda needs to know whether to ignore them or log them. | Lambda processing | ML |
| 3 | **`avg_memory_utilization` removed from model** — this metric currently has a pre-built Slack Canvas (`demo_canvas.py`) and a Windows article. Confirm: is it fully dropped, or kept as a static rule (e.g. always flag if memory > X%)? | Canvas demo, article routing | Data / ML |
| 4 | **Windows articles for `avg_boot_time` and `p90_boot_time`** — these files don't exist yet (`windows_avg_boot_time.md`, `windows_p90_boot_time.md`). Users flagged for these metrics will see fallback text. Need content from IT/data team. | User experience | IT / Content |
| 5 | **SageMaker `device` field** — confirm the telemetry CSV uses a hostname that includes `"hp"` or `"windows"` (case-insensitive). If not, the bot will serve Mac articles to Windows users. | Article routing | Backend / Data |
| 6 | **S3 bucket name and IAM permissions** — Lambda needs `s3:GetObject` on the telemetry bucket and `sagemaker:InvokeEndpoint` on the model endpoint. Who configures IAM? | Infrastructure | DevOps |
| 7 | **Alert deduplication** — if Lambda is triggered multiple times for the same user within a short window (e.g. multiple files in one day), the user receives duplicate DMs. Where does deduplication live — Lambda or the bot? | User experience | Backend |
| 8 | **Opt-out storage for multi-instance deployment** — `data/opt_outs.json` is a local file. If the bot runs on multiple instances or restarts, opt-outs may be lost. Needs a shared store (DynamoDB, RDS, etc.) before production. | Reliability | Backend |
| 9 | **Lambda → bot invocation method** — Lambda currently calls `send_alert()` by importing the module directly (assumes co-located). If the Slack bot runs as a separate service, Lambda needs an HTTP endpoint or message queue. Architecture not yet decided. | Deployment topology | Backend |
