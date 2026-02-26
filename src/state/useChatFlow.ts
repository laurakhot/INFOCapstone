import { useEffect, useRef, useState } from 'react';
import type {
  CaseData,
  ThinkingState,
  DiagnosticState,
  ThinkingStepItem,
  DiagnosticRowItem,
} from '@/types/chat.types';
import { useConversation } from './conversationStore';
import {
  buildTextMessage,
  buildChipsMessage,
  buildRootCauseMessage,
  buildResolutionMessage,
  buildEscalationMessage,
  buildTimeSeriesMessage,
} from '@/services/chatService';
import { runPrediction } from '@/services/predictionService';
import type { DeviceSnapshot } from '@/types/api.types';
import { escapeHtml } from '@/utils/formatters';
import { validateDeviceSnapshotJson } from '@/utils/validators';

function delay(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

export function useChatFlow(caseData: CaseData) {
  const { state, dispatch } = useConversation();
  const cancelRef = useRef(false);
  const triedChipsRef = useRef<Set<string>>(new Set());
  const startChipsMsgIdRef = useRef<string>('');

  const [thinkingState, setThinkingState] = useState<ThinkingState | null>(null);
  const [diagnosticState, setDiagnosticState] = useState<DiagnosticState | null>(null);
  const [flowDisabled, setFlowDisabled] = useState(false);

  // Cancel in-flight animation and reset when the scenario (case) switches
  useEffect(() => {
    cancelRef.current = true;
    triedChipsRef.current = new Set();
    startChipsMsgIdRef.current = '';
    setThinkingState(null);
    setDiagnosticState(null);
    setFlowDisabled(false);

    // Allow a tick for state to flush before new flow can start
    const t = setTimeout(() => {
      cancelRef.current = false;
    }, 50);
    return () => {
      clearTimeout(t);
      cancelRef.current = true;
    };
  }, [caseData.id]);

  // ─── Badge swipe ────────────────────────────────────────────────────────────

  async function handleBadgeSwipe() {
    cancelRef.current = false;
    dispatch({ type: 'SET_PHASE', phase: 'active' });

    await delay(300);
    if (cancelRef.current) return;

    dispatch({
      type: 'ADD_MESSAGE',
      message: buildTextMessage(
        `<p>${escapeHtml(caseData.greeting)}, <strong>${escapeHtml(caseData.user)}</strong> 👋</p>` +
          `<p>I can see your device: <strong>${escapeHtml(caseData.device)}</strong>. ` +
          `Let me run a quick health check.</p>`
      ),
    });

    await delay(700);
    if (cancelRef.current) return;

    const startMsg = buildChipsMessage([
      { id: 'start', label: '🔍 Start diagnosis', primary: true },
    ]);
    startChipsMsgIdRef.current = startMsg.id;
    dispatch({ type: 'ADD_MESSAGE', message: startMsg });
  }

  // ─── Diagnosis flow ─────────────────────────────────────────────────────────

  async function runDiagnosis() {
    setFlowDisabled(true);

    if (startChipsMsgIdRef.current) {
      dispatch({ type: 'MARK_CHIPS_USED', messageId: startChipsMsgIdRef.current });
    }

    // --- Thinking animation ---
    const thinkingSteps: ThinkingStepItem[] = caseData.thinkingSteps.map((text) => ({
      text,
      visible: false,
      done: false,
    }));
    setThinkingState({ steps: thinkingSteps, fading: false });

    for (let i = 0; i < thinkingSteps.length; i++) {
      await delay(i === 0 ? 400 : 1300);
      if (cancelRef.current) return;

      setThinkingState((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          steps: prev.steps.map((s, j) => ({
            ...s,
            visible: j <= i,
            done: j < i,
          })),
        };
      });
    }

    // Mark last step done
    await delay(900);
    if (cancelRef.current) return;
    setThinkingState((prev) =>
      prev ? { ...prev, steps: prev.steps.map((s) => ({ ...s, done: true })) } : prev
    );

    await delay(500);
    if (cancelRef.current) return;

    // Fade out thinking block
    setThinkingState((prev) => (prev ? { ...prev, fading: true } : prev));
    await delay(480);
    if (cancelRef.current) return;
    setThinkingState(null);

    // --- Diagnostic rows animation ---
    const diagRows: DiagnosticRowItem[] = caseData.diagnosticRows.map((row) => ({
      ...row,
      visible: false,
    }));
    setDiagnosticState({ rows: diagRows, fading: false });

    for (let i = 0; i < diagRows.length; i++) {
      await delay(i === 0 ? 200 : 700);
      if (cancelRef.current) return;

      setDiagnosticState((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          rows: prev.rows.map((r, j) => ({ ...r, visible: j <= i })),
        };
      });
    }

    await delay(900);
    if (cancelRef.current) return;

    // Fade out diagnostic block
    setDiagnosticState((prev) => (prev ? { ...prev, fading: true } : prev));
    await delay(480);
    if (cancelRef.current) return;
    setDiagnosticState(null);

    // --- Emit results ---
    // In non-demo mode, call the real API gateway and attach the raw response for debug display.
    let apiDebug: Parameters<typeof buildRootCauseMessage>[1];
    try {
      const result = await runPrediction(caseData.snapshot as unknown as DeviceSnapshot, caseData);
      apiDebug = result.rawResponse?.prediction;
    } catch {
      // Swallow — UI still renders from caseData mock even if API fails
    }
    dispatch({ type: 'ADD_MESSAGE', message: buildRootCauseMessage(caseData, apiDebug) });

    await delay(500);
    if (cancelRef.current) return;

    switch (caseData.resolution) {
      case 'self_service': {
        dispatch({
          type: 'ADD_MESSAGE',
          message: buildChipsMessage(caseData.chips),
        });
        break;
      }

      case 'it_support_desk':
      case 'hardware_upgrade': {
        // Case 3: show 30-day time series before escalation card
        if (caseData.timeSeries && caseData.timeSeries.length > 0) {
          const p75Ram =
            caseData.benchmarkSummary.metrics.find((m) => m.label === 'RAM Usage')
              ?.p75Value ?? 70;
          dispatch({
            type: 'ADD_MESSAGE',
            message: buildTimeSeriesMessage(caseData.timeSeries, p75Ram),
          });
          await delay(500);
          if (cancelRef.current) return;
        }

        dispatch({
          type: 'ADD_MESSAGE',
          message: buildEscalationMessage(
            'in_person',
            [],
            caseData.directEscalationChips ?? []
          ),
        });
        break;
      }
    }

    setFlowDisabled(false);
  }

  // ─── Chip clicks ────────────────────────────────────────────────────────────

  async function handleChipClick(chipId: string) {
    if (chipId === 'start') {
      await runDiagnosis();
      return;
    }

    // Self-service action chip
    const resolution = caseData.responses[chipId];
    if (!resolution) return;

    setFlowDisabled(true);

    const chipLabel = caseData.chips.find((c) => c.id === chipId)?.label ?? chipId;
    dispatch({
      type: 'ADD_MESSAGE',
      message: buildTextMessage(`<p>${escapeHtml(chipLabel)}</p>`, 'user'),
    });

    await delay(400);
    if (cancelRef.current) return;

    dispatch({ type: 'ADD_MESSAGE', message: buildResolutionMessage(resolution) });
    triedChipsRef.current.add(chipId);

    await delay(800);
    if (cancelRef.current) return;

    const remaining = caseData.chips.filter((c) => !triedChipsRef.current.has(c.id));
    dispatch({
      type: 'ADD_MESSAGE',
      message: buildEscalationMessage('self_service', remaining, []),
    });

    setFlowDisabled(false);
  }

  async function handleDirectChipClick(chipId: string) {
    const resolution = caseData.responses[chipId];
    if (!resolution) return;
    dispatch({ type: 'ADD_MESSAGE', message: buildResolutionMessage(resolution) });
  }

  // ─── GSD escape hatch ───────────────────────────────────────────────────────

  async function handleGsdEscape() {
    dispatch({
      type: 'ADD_MESSAGE',
      message: buildEscalationMessage('gsd', [], []),
    });
    setFlowDisabled(true);
  }

  // ─── Free-text input ────────────────────────────────────────────────────────

  async function handleSend(text: string) {
    if (flowDisabled) return;

    dispatch({
      type: 'ADD_MESSAGE',
      message: buildTextMessage(`<p>${escapeHtml(text)}</p>`, 'user'),
    });

    const lower = text.toLowerCase();
    const wantsHuman =
      lower.includes('human') ||
      lower.includes('agent') ||
      lower.includes('person') ||
      lower.includes('escalat');

    await delay(600);
    if (cancelRef.current) return;

    if (wantsHuman) {
      await handleGsdEscape();
    } else {
      dispatch({
        type: 'ADD_MESSAGE',
        message: buildTextMessage(
          "<p>Happy to help. If you'd like to connect with a GSD agent at any time, just ask.</p>"
        ),
      });
    }
  }

  // ─── File upload ────────────────────────────────────────────────────────────

  async function handleFileSelect(file: File) {
    dispatch({
      type: 'ADD_MESSAGE',
      message: buildTextMessage(
        `<p>📎 Processing snapshot: <strong>${escapeHtml(file.name)}</strong>…</p>`
      ),
    });

    try {
      const result = await validateDeviceSnapshotJson(file);
      if (!result.valid) {
        dispatch({
          type: 'ADD_MESSAGE',
          message: buildTextMessage(
            `<p>⚠️ Could not process file: ${result.errors.map(escapeHtml).join('; ')}</p>`,
            'ai',
            { isError: true }
          ),
        });
        return;
      }

      dispatch({
        type: 'ADD_MESSAGE',
        message: buildTextMessage(
          '<p>✅ Snapshot validated. In production this triggers the full prediction pipeline.</p>'
        ),
      });
    } catch {
      dispatch({
        type: 'ADD_MESSAGE',
        message: buildTextMessage('<p>⚠️ Failed to read file.</p>', 'ai', { isError: true }),
      });
    }
  }

  return {
    thinkingState,
    diagnosticState,
    // Disable input during animation and while on badge screen
    flowDisabled: flowDisabled || state.phase === 'badge',
    handleBadgeSwipe,
    handleChipClick,
    handleDirectChipClick,
    handleGsdEscape,
    handleSend,
    handleFileSelect,
  };
}
