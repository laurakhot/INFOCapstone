# Husky IT Support Kiosk — Demo App

React 18 + TypeScript + Vite demo of an AI-powered IT support kiosk.

## Quick start

```bash
npm install
npm run dev
```

Open `http://localhost:5173` — the app runs at full 16:9 aspect ratio.

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

## Build

```bash
npm run build   # outputs to dist/
npm run preview # serve the production build locally
```
