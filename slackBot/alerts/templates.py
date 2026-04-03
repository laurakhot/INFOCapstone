"""
Block Kit message builders for laptop health alerts.
"""

# Maps metric keys to human-friendly display labels
METRIC_LABELS = {
    "avg_memory_utilization": "Memory Usage",
    "avg_boot_time": "Startup Speed",
    "max_memory_pressure": "System Performance",
    "cpu_count": "Processing Capacity",
    "battery_cycle": "Battery Health",
}

# Metrics that require hardware replacement (user cannot self-fix)
REPLACEMENT_METRICS = {"cpu_count", "battery_cycle"}


def build_alert_blocks(
    first_name: str,
    device: str,
    predictions: dict,
) -> list:
    """
    Build Block Kit blocks for the initial alert DM.

    predictions: dict mapping metric_key -> confidence float (0.0 - 1.0)
    Returns at most 3 root causes above 50% confidence.
    """
    # Filter and sort by confidence, take top 3
    risks = sorted(
        [(k, v) for k, v in predictions.items() if v > 0.5],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    if not risks:
        return []

    # Build the bulleted risk list text
    risk_lines = []
    for metric_key, confidence in risks:
        label = METRIC_LABELS.get(metric_key, metric_key)
        pct = int(confidence * 100)
        risk_lines.append(f"• *{label}*    {pct}% likely contributing")
    risk_text = "\n".join(risk_lines)

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Hey {first_name} 👋 Our system detected some risk factors that may "
                    f"cause your laptop to crash.\n\n"
                    f"Here's what we found on *{device}*:\n\n"
                    f"{risk_text}"
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "I can help you with these! Which one would you like to tackle first?",
            },
        },
        # Root cause buttons (one per risk)
        {
            "type": "actions",
            "block_id": "root_cause_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": METRIC_LABELS.get(k, k)},
                    "action_id": f"root_cause__{k}",
                    "value": k,
                }
                for k, _ in risks
            ],
        },
        # Snooze + opt-out buttons
        {
            "type": "actions",
            "block_id": "alert_management_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Remind me later"},
                    "action_id": "snooze",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Don't remind me again"},
                    "action_id": "opt_out",
                },
            ],
        },
    ]

    return blocks


def build_article_blocks(article_mrkdwn: str, metric_key: str) -> list:
    """
    Build Block Kit blocks to display a KB article with a resolution button.
    """
    is_replacement = metric_key in REPLACEMENT_METRICS

    if is_replacement:
        button_text = "✅ Got it, I'll back up my data"
        action_id = f"done_replacement__{metric_key}"
    else:
        button_text = "✅ Done, re-run the diagnosis"
        action_id = f"done_selfservice__{metric_key}"

    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": article_mrkdwn},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Once you've gone through the steps, I can run a fresh check "
                    "to confirm your laptop is back to normal!"
                    if not is_replacement
                    else "When you're ready, let me know!"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": f"resolution_actions__{metric_key}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": button_text},
                    "action_id": action_id,
                    "value": metric_key,
                    "style": "primary",
                }
            ],
        },
    ]

    return blocks


def build_acknowledged_blocks(risk_labels: list) -> list:
    """Replacement for the original alert after acknowledge."""
    labels_text = " · ".join(risk_labels)
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"✅ *Issue resolved*\n_{labels_text}_",
            },
        }
    ]


def build_snoozed_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "⏰ *Snoozed* — I'll check back with you in an hour. Take care 😊",
            },
        }
    ]


def build_opted_out_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "🔕 *You've unsubscribed from alerts like this.*",
            },
        }
    ]


def build_opt_out_feedback_blocks() -> list:
    """Ask for optional opt-out feedback."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Got it. You've unsubscribed from alerts like this. 👍\n\n"
                    "Just out of curiosity, what made you decide to opt out? "
                    "_(Feel free to skip — this is totally optional 😊)_"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "opt_out_feedback_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Seems like a false positive"},
                    "action_id": "opt_out_reason__false_positive",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Already replacing my laptop"},
                    "action_id": "opt_out_reason__replacing",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Getting too many alerts"},
                    "action_id": "opt_out_reason__too_many",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Skip"},
                    "action_id": "opt_out_reason__skip",
                },
            ],
        },
    ]
