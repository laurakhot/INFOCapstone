"""
App Home view builder.

Published when a user opens the bot's App Home tab.
"""


def build_home_view() -> dict:
    return {
        "type": "home",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Laptop Health Assistant 💻",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "Hi there! 👋 I keep an eye on your laptop's health and give you "
                        "a heads-up *before* things go wrong — so you can fix small issues "
                        "before they turn into a crash or data loss."
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*What I monitor* 🔍",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": "🧠 *Memory Usage*\nHigh RAM consumption that can slow down or crash your Mac"},
                    {"type": "mrkdwn", "text": "⚡ *System Performance*\nMemory pressure that strains your system"},
                    {"type": "mrkdwn", "text": "🚀 *Startup Speed*\nSlow boot times that signal underlying issues"},
                    {"type": "mrkdwn", "text": "🔋 *Battery Health*\nDegradation that may cause unexpected shutdowns"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*How it works* ⚙️",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "1. Our system continuously analyzes your laptop's telemetry data\n"
                        "2. If a risk is detected, I'll send you a *direct message* with details\n"
                        "3. You'll get step-by-step guidance to resolve it — or a heads-up to back up your data\n"
                        "4. Once you've acted, I'll re-check to confirm everything is healthy"
                    ),
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Got a question?* 💬",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "Just send me a message in the *Messages* tab — I can explain any "
                        "alert you've received or walk you through troubleshooting steps.\n\n"
                        "_Try asking: \"What does memory pressure mean?\" or \"How do I speed up my startup?\"_"
                    ),
                },
            },
        ],
    }
