# Laptop Issue Prediction with Device Metrics for Amazon IT

Husky Support

Meiyao Li, Kira Brodsky, Thea Klein\-Balajee, Laura Khotemlyanky, Lexeigh Kolakowski

# Description

**Software**

We built a machine learning model that takes in telemetry data from an Amazonian&#39;s device and outputs a prediction, with a confidence level, of what hardware issue the user is likely experiencing\. The model surfaces up to three root causes ranked by confidence score, covering memory utilization, boot time, system performance pressure, CPU capacity, and battery health\.

**UX**

We built a Slack chatbot \(Husky Support\) that delivers proactive, non\-technical hardware alerts directly to Amazonians via DM\. When the telemetry backend detects risk, the bot sends a warm, plain\-language message listing the top contributing factors with confidence percentages and interactive buttons that lead to resolution steps\. Users can explore these steps, confirm a fix and trigger a fresh diagnosis, snooze the alert, or opt out, all without leaving Slack or contacting IT\.

# Problem

**How might we support Amazonians through proactive measures so that an unexpected hardware failure can be avoided or mitigated\.**

# Why

Amazonians do not want to lose work, miss deadlines, or interrupt their day waiting for a fix that could have been prevented\. Specifically, three pain points came through clearly:

**1\. Predictability over disruption\.** Amazonians, especially L8\+ leaders, treat their laptop as the primary vessel for organizational work\. Unexpected crashes risk losing data that hasn&#39;t been saved, interrupting high\-stakes tasks, and eroding trust in the IT department\.

**2\. Reactive to Predictive IT Support\.** Traditional IT support is reactive: a crash happens, a ticket opens, a replacement is applied\. But most crashes are preceded by measurable signals: rising boot times, elevated memory pressure, degraded battery cycles\. Our system detects these signals early and surfaces them to the end users before failure occurs\.

**3\. Respecting Amazonians&#39; time by accurate routing\.** 

When Amazonians attempt self\-service solutions but discover the issue is beyond their control, they hit a frustrating dead end\. They are then required to reach out to an ITSE and walk them through their issue and troubleshooting steps, wasting time explaining everything they had just done\.

Our research showed end users want one of two things: either fix it themselves quickly, or skip straight to an ITSE without wasted steps\. The cost of an unresolved issue is twofold: explicit \(replacement hardware\) and implicit \(lost productivity\)\. By knowing upfront whether an issue is self\-serviceable or requires replacement, the bot routes users to the right path immediately\.

# Success

\[Outcome 1\]: Proactive crash prevention \(alert users of imminent crash and facilitate resolution\)\[Outcome 2\]: Accurate routing \(users are navigated to the appropriate resolution, whether that be an in person ITSE, online ITSE, or self\-service articles\)

# Audience

**Primary — Amazonians with imminent hardware failure**  Any Amazonian whose laptop is showing early warning signals or has already encountered a failure\. They interact with our system directly in Slack, receive a plain\-language explanation of what&#39;s wrong, guided steps to fix it themselves, or a clear handoff to IT when self\-service isn&#39;t an option\. They don&#39;t need to open a ticket, repeat their issue, or know anything about the underlying hardware to get value\.

**Secondary — IT Support Engineers \(ITSEs\)** Engineers who handle hardware failure support cases, rather than starting every case from scratch, they receive a pre\-diagnosed device snapshot with ranked root causes and confidence scores before the conversation begins\. This shorten time\-to\-resolution and helping them consistently hit the 23\-minute average handling time target\.

# Scope of work

## Must have features \(P0\)

**Data \(**@Lexeigh @Thea Klein-Balajee Please proofread and make updates\):

- **Random forest model** trained on 90th percentile telemetry data that predicts **imminent system failure such as BSOD &#x2F; black screen &#x2F; kernel crash\.**
- **Feature\-level prediction models** — 12 separate models, each trained on a single feature, with outputs combined into a single computer\-wide diagnosis

**Backend \(**@Laura Khotemlyansky Please proofread and make updates\):

- A backend that use EventBridge \+ Lambda cron to identify at\-risk users from a S3 bucket with telemetry snapshot data\.
    - \(For demo, our backend is On\-Demand &#x2F; Offline\. It scans a static S3 bucket and will be manually triggered\.\)

**Slack Chatbot Supporting End Users \(UX\)**

- DM at\-risk users on Slack workspace via Socket Mode with predictions from the model\.
- Chatbot is able to follow 3 core user flows to demonstrate potential use cases:
    - **Self\-serviceable:** End users receive Slack notifications that laptop might crash &#x2F; BSOD in an hour\. Root cause predictions are avg\_memory\_utilization, avg\_boot\_time, and max\_memory\_pressure\. User self service and prevent hardware failure\.
    - **Not\-self Serviceable:** End users receive Slack notifications, root cause predictions are cpu\_count, battery cycle\. User cannot solve it by themselves, but they can backup data &amp; say farewell to their laptops before seeking replacement in person\.
    - **Contact Deflection**: People who worry they have a laptop failure due to unresponsive laptop can ask Slack on mobile\. If top 3 prediction has low importance, their laptop is safe\.

## Should have and nice\-to\-have features \(P1&#x2F;P2\)

**ITSE summary of Slack interaction**

- Provide ITSEs with a summary of the interaction between the Amazonian and the chatbot so that the ITSE doesn&#39;t need to waste time context gathering\.
- Generate:
    - Likely issue category \(software &#x2F; OS &#x2F; hardware\)
    - Confidence level and alternative hypotheses
    - Suggested next diagnostic steps

**Multiple Potential Root Causes Surfaced**

- Main potential diagnosis, and other potential diagnoses alongside it with confidence scores to raise multiple concerns to user or ITSE and identify other potential future concerns

**Clear Resolution Path Recommendation**

- Assist non\-tech Amazonians with device issues by providing them clear direction and prediction with confidence
- Assist ITSE in deciding between recovery, guided fix, or replacement\.

## Technologies

- Next\.JS, Figma, Python, Slack API

## Deliverables

- Working model that can predict IT issues based on data provided
- Interactive interface via Slack for Amazonians with device issues to receive information from the model
- User research report of low fidelity concept validation

## Features that aren&#39;t in scope

- Solving for any confidential metrics \(network, VPN, or SSO issues\)
- Solving for specific app crashes&#x2F;breakdowns
- Solving for physical laptop issues \(e\.g\. cracked screen, etc\)

