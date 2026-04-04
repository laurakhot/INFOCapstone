"""
Block Kit message builders for laptop health alerts.
"""

# Maps metric keys to human-friendly display labels (used in text/summaries)
METRIC_LABELS = {
    "max_cpu_usage": "Max CPU Usage",
    "avg_memory_utilization": "Average Memory Utilization",
    "cpu_count": "Processing Capacity",
    "memory_size_gb": "Memory Capacity",
    "p90_boot_time": "Worst Boot Time",
    "uptime_days": "System Uptime",
}

# Actionable button labels shown on the root cause buttons
METRIC_BUTTON_LABELS = {
    "max_cpu_usage": "Optimize CPU Usage",
    "avg_memory_utilization": "Optimize Memory",
    "cpu_count": "Protect my Data",
    "memory_size_gb": "Protect my Data",
    "p90_boot_time": "Speed Up Boot Time",
    "uptime_days": "Restart my Laptop",
}


def build_alert_blocks(
    first_name: str,
    device: str,
    predictions: dict,
    has_replacement: bool = False,
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

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Hey {first_name} 👋 Your laptop is showing signs of hardware wear "
                    f"that could lead to unexpected data loss."
                    if has_replacement else
                    f"Hey {first_name} 👋 Our system detected some risk factors that may "
                    f"cause your laptop to crash."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Here's what we found on your *{device}*:",
            },
        },
        # aesthetic — card-style risk list; primary root cause gets 🔴, secondary gets 🟡
        *[
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"{'🔴' if i == 0 else '🟡'} *{METRIC_LABELS.get(k, k)}*",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"{int(v * 100)}% likely contributing",
                    },
                ],
            }
            for i, (k, v) in enumerate(risks)
        ],
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "Let me help you protect your data before it's too late."
                    if has_replacement else
                    "Do you want me to walk you through how to prevent it?"
                ),
            },
        },
        {
            "type": "actions",
            "block_id": "root_cause_actions",
            "elements": [
                # Replacement: fixed process actions (not metric-mapped)
                *([
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Backup my Data"},
                        "action_id": "replacement_action__data_backup",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Request Laptop Replacement"},
                        "action_id": "replacement_action__replacement",
                    },
                ] if has_replacement else [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": METRIC_BUTTON_LABELS.get(k, METRIC_LABELS.get(k, k))},
                        "action_id": f"root_cause__{k}",
                        "value": k,
                    }
                    for k, _ in risks
                ]),
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


def build_article_blocks(article_mrkdwn: str, metric_key: str, is_replacement: bool = False) -> list:
    """
    Build Block Kit blocks to display a KB article with a resolution button.
    is_replacement should be passed by the caller (from rag.articles.is_replacement()).
    """
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


def build_still_at_risk_blocks(remaining: dict) -> list:
    """
    Compact follow-up shown after re-diagnosis when risk factors remain.
    No introductory text — just the updated action buttons.
    """
    risks = sorted(
        [(k, v) for k, v in remaining.items() if v > 0.5],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    if not risks:
        return []

    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "I'm still seeing some risk factors. Which would you like to tackle next?",
            },
        },
        {
            "type": "actions",
            "block_id": "root_cause_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": METRIC_BUTTON_LABELS.get(k, METRIC_LABELS.get(k, k))},
                    "action_id": f"root_cause__{k}",
                    "value": k,
                }
                for k, _ in risks
            ],
        },
    ]
