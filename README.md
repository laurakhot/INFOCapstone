Problem Context:

Modern enterprise IT systems generate signals at the software, operating system, and hardware layers. When failures occur, these systems often leave behind actionable artifacts - crash logs, dump files, error codes, and device telemetry - that can precisely identify root causes and speed up the process of IT engineers diagnosing problems or users mitigating problems by themselves without visiting the IT office. However, the triaging telemetry has not surfaced to support engineers or Amazonians. The system should enable both self-service resolution and efficient escalation when human intervention is required.


The codebase currently includes past example work for the team to reference. And the code for a few tests of models. In the future this repo will also include the code to the website that the user will be able to interact with. We are currently keeping the data out of the repo.


To build the code you will have to clone the repo, install the needed packages, and import data to be analyzed. To contribute to the code, you only need to write code locally and push it into the repo.

# Husky IT Support Kiosk — Demo App

React 18 + TypeScript + Vite demo of an AI-powered IT support kiosk.


## Three scenarios

| Toggle | Use case | Resolution path |
|--------|----------|-----------------|
| Case 1 | macOS Slow — Resource Optimization | Self-service chips (restart, RAM, disk) |
| Case 2 | Windows Slow — Hardware Failure (EOL) | Skip chips → in-person IT support |
| Case 3 | Windows Slow — Hardware Insufficiency | 30-day time series + upgrade request |

Each case loads a **realistic device snapshot JSON** from `src/data/` and runs it through the same pipeline a real uploaded snapshot would follow.

### Flow (all cases)
1. Click the badge screen to simulate a badge swipe
2. Click **Start diagnosis** to begin the animated prediction chain
3. Watch the thinking block and diagnostic table appear and fade out
4. Root cause analysis appears with ranked confidence bars and P75 benchmark

## Architecture

```
src/
├── types/          chat.types.ts · api.types.ts · globals.d.ts
├── data/           case1.json · case2.json · case3.json · index.ts
├── services/       apiClient.ts · predictionService.ts · chatService.ts
├── state/          conversationStore.tsx (Context + useReducer) · useChatFlow.ts
├── utils/          validators.ts · formatters.ts
└── components/     16 components, each with CSS Module
```

### Demo vs. production mode

Set `VITE_DEMO_MODE=false` in `.env.local` to enable real API calls:

```
POST /api/device-snapshot  →  { prediction_id }
GET  /api/prediction/:id   →  { status, root_cause, confidence_score }
```

In demo mode (`VITE_DEMO_MODE=true`, the default), `predictionService` returns the
case JSON's pre-computed root causes — no network calls needed.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `/api` | Backend base URL |
| `VITE_DEMO_MODE` | `true` | Skip API; use local case data |

Copy `.env.example` to `.env.local` and adjust as needed.