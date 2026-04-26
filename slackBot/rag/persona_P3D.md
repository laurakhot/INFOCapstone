# Persona: P3-D

## Summary
This persona represents a non-technical employee using a company-managed laptop and receiving proactive IT support notifications through a workplace messaging tool.

He is capable of following clear instructions, but does not want to troubleshoot open-ended technical issues unless the path is simple, trustworthy, and clearly worth his time.

He values speed, clarity, autonomy, and reassurance. He wants to know:
- what is happening,
- whether it matters,
- what he should do next,
- and whether the issue is his to handle or IT's to own.

He is not resistant to self-service. He is resistant to ambiguity, excess effort, and technical jargon.

---

## Archetype
**Role archetype:** Non-technical or lightly technical knowledge worker
**Environment:** Company-managed laptop, IT-supported workplace, messaging-based workflow
**Technical confidence:** Moderate in everyday use, low in systems troubleshooting
**Primary goal:** Stay productive and resolve issues with minimal disruption
**Secondary goal:** Do due diligence before escalating to IT

---

## Core Mindset
He does not think of the support bot as a "chatbot" first. He thinks of it as a **notification and routing system** that should tell him exactly what to do when something needs attention.

He expects:
- direct guidance over exploration,
- predefined answers over generative ambiguity,
- action-first communication over explanation-first communication,
- and a clear escalation path when self-service is no longer appropriate.

---

## Mental Model

### 1) "Tell me what to do first"
His first question is not "why is this happening?"
It is: **"What do I do right now?"**

He prefers:
1. immediate action,
2. then explanation,
3. then optional extra detail.

If a message leads with too much context, he experiences it like a "recipe blog" when he only wanted the ingredients.

### 2) "Trust comes from structure"
He trusts a structured, predefined guide from IT more than an open-ended conversation.
A canvas/article feels official, stable, and repeatable. A chatbot feels less predictable.

### 3) "I only care about technical details if they help me act"
Raw metrics like CPU %, memory %, or uptime days do not matter on their own.
They only become useful when translated into:
- consequence,
- urgency,
- threshold,
- and recommended action.

### 4) "I want due diligence without becoming IT"
He is willing to try simple, trustworthy self-service steps.
That gives him a sense of agency and helps him avoid wasting IT's time.
But he does not want to become responsible for complex diagnosis or risky repair.

### 5) "Ownership should be obvious"
He wants every issue to clearly signal one of two things:
- **I can handle this**
- **IT needs to handle this**

If that line is blurry, he becomes uncertain and less likely to act confidently.

---

## Typical Workday

His day is full of meetings, tabs, messages, and task-switching. He is often busy, mid-flow, or context-loaded when a support notification appears.

- Multitasking is his default state — a notification competes with everything else on his screen
- He may be in a meeting or mid-document when an alert arrives
- He often does not have time to troubleshoot immediately and needs to defer intelligently, not ignore irresponsibly
- He is optimizing for getting back to work quickly, not understanding the full picture
- Unlike P2-M, he is less likely to bring deep context to IT — he wants the system to carry that context for him

**Implication for design:** The system must support quick comprehension, fast decision-making, low cognitive load, and easy snooze behavior without losing important context.

---

## IT Support Behavior & Habits

### What he will do
- Follow simple step-by-step instructions if they look safe and official
- Try easy fixes first when the expected outcome is visible
- Escalate when the issue is clearly outside his scope
- Re-check a fix if the system helps him confirm it worked

### What he will avoid
- Reading dense walls of text
- Interpreting technical numbers without explanation
- Choosing among several parallel troubleshooting options with no recommended order
- Using open-ended chat as the main way to solve a problem
- Taking actions that feel risky, irreversible, or outside his role

### What builds his confidence
- "Do this first" — a single, ordered next step
- Explicit thresholds stated in plain language
- Visible current-vs-recommended state
- Clear success/failure feedback after he acts
- Reassurance when escalation is the right path, not a failure

### What erodes his confidence
- Jargon without consequence ("your p90 CPU temp is elevated")
- Arbitrary risk labels without context ("risk factor detected")
- Too many choices at once with no priority
- Alerts that feel alarmist but vague
- Conversational UX that makes the resolution feel unpredictable

---

## Decision Rationale

| Situation | What he does | Why |
|---|---|---|
| Receives a vague notification | Likely dismisses or defers | No clear action = not worth interrupting his flow |
| Told issue takes 5 min to fix | Likely follows the steps | Low time cost, clear ROI |
| Steps look complex or unclear | Stops and escalates to IT | Does not want to risk making things worse |
| Told IT needs to handle it | Wants a clear next step and timeline | Needs to plan; open-ended "contact IT" is not enough |
| Fix appears to work | Wants confirmation from the system | Needs closure; won't self-assess whether the laptop is actually better |
| Alert arrives mid-meeting | Snoozes until he can give it attention | Needs defer-without-losing-context, not permanent opt-out |

---

## What Works for Him

- **Action-first messaging** — lead with what to do, explain why second
- **Numbered steps with one clear starting point** — no parallel options without priority
- **Outcome language over metric language** — "your laptop could crash soon" > "87% memory utilization"
- **Ownership clarity** — every alert should make it immediately clear: self-fix or IT's job
- **Confirmation after action** — re-run diagnosis flow tells him whether the fix worked
- **Snooze to tomorrow** — "end of day" is too soon when he's in back-to-back meetings
- **Canvas/article as the answer** — structured, official-feeling guides feel more trustworthy than chatbot responses

## What Doesn't Work for Him

- Opening with explanation before action (he will skim past it)
- Raw percentages or metric names with no translation into consequences
- Multiple fix options presented without a recommended sequence
- Ambiguous ownership ("here are some things you could try")
- Permanent opt-out on high-severity alerts — he may not realize he silenced something important
- Alerts that don't confirm resolution — he won't know if he actually fixed it

---

## UX Principles to Validate Ideas Against

### Clarity
- Does this tell him what to do within a few seconds of reading?
- Is the next step obvious without needing to scroll or think?

### Trust
- Does this feel predefined, stable, and from IT — not generated on the fly?
- Are the steps concrete enough that he can follow them without interpretation?

### Relevance
- Are details tied to consequence and action, not just system state?
- Is there any information here he doesn't need right now?

### Control
- Does he feel in control of the next step, even if it's "contact IT"?
- Can he defer without losing the thread?

### Ownership
- Is it immediately obvious whether he should handle this or IT should?

### Cognitive load
- Does this reduce decisions rather than add them?
- Are we guiding a sequence rather than presenting options all at once?

### Validation
- Can he tell whether the fix worked?
- Does the system close the loop, or does he have to judge for himself?

---

## Anti-Patterns

Do not validate ideas that:
- start with explanation instead of action,
- rely on raw numbers alone without consequence framing,
- require the user to infer severity or ownership,
- present parallel fix options without a recommended order,
- blur the line between self-service and IT-owned work,
- make chat the primary interface for first-line troubleshooting,
- or let users suppress important alerts too easily.

---

## Contrast with P2-M

| Dimension | P2-M | P3-D |
|---|---|---|
| Technical confidence | Moderate-high, attempts diagnosis | Low, wants a predefined path |
| First response to an alert | Tries to self-resolve immediately | Evaluates effort cost before acting |
| Context brought to IT | Full — device, symptoms, timeline, prior steps | Minimal — expects the system to carry context |
| Trust mechanism | Personalization and specificity | Structure, sequence, and official source |
| Urgency signal | Felt experience (crashes, freezes) | Stated consequence and timeframe |
| Tolerance for ambiguity | Low but pushes through | Very low — stops at ambiguity |

---

## One-Line Design Rule
**If he cannot tell within a few seconds what is happening, whether he should care, and what he should do next, the design is not ready.**
