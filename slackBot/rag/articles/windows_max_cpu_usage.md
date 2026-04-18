# 🔍 Find who's causing CPU Spike

## Resolution Steps:
1. **Open Task Manager:** Press `Ctrl + Shift + Esc`.
2. **Sort by CPU:** Click the **Processes** tab and clickc**CPU** the column header.
3. **Close the culprit** if it's something you don't need right now (a stuck browser tab, a background sync, an app you forgot was running).
4. **Force-end unresponsive processes:** right-click the process → **End task**. Use this when an app is frozen or won't close normally.

## What's happening
Your CPU is spiking because a specific process is doing heavy work right now — not because your machine is overloaded overall. Baseline memory is low, which means apps aren't piling up; instead, one process is burning cycles in bursts.

Common culprits on Windows: antivirus or security agent scans (Defender, CrowdStrike, Zscaler), Windows Update, Search Indexer, OneDrive syncing, or a video call encoding in the background. These are usually temporary — if the spike passes on its own, no action needed. If it stays pinned, end the process or restart it.