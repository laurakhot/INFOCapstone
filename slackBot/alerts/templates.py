"""
Block Kit message builders for laptop health alerts.
"""

from typing import Optional

METRIC_LABELS = {
    "avg_processor_time":     "Processor Time",
    "max_cpu_usage":          "Max CPU Usage",
    "avg_memory_utilization": "Average Memory Utilization",
    "avg_battery_health":     "Battery Health",
    "uptime_days":            "System Uptime",
    "p90_cpu_temp":           "CPU Temperature",
}

METRIC_BUTTON_LABELS = {
    "avg_processor_time":     "Check Processor Performance",
    "max_cpu_usage":          "Reduce CPU Load (3 min)",
    "avg_memory_utilization": "Reduce Memory Usage (3 min)",
    "avg_battery_health":     "Check Battery Health",
    "uptime_days":            "Restart your Laptop (2 min)",
    "p90_cpu_temp":           "Cool Down your Laptop",
}

# Healthy threshold per metric. Flagged when value exceeds threshold
# (or falls below, for avg_battery_health).
METRIC_THRESHOLDS = {
    "avg_memory_utilization": 78.1,
    "uptime_days":            27,
    "avg_processor_time":     67.6,
    "max_cpu_usage":          90.2,
    "avg_battery_health":     93.7,  # flagged when BELOW this
    "p90_cpu_temp":           88.1,
}

# Metrics where a lower value is worse (threshold is a floor, not a ceiling).
METRIC_LOWER_IS_WORSE = {"avg_battery_health"}

# Format-string templates. Placeholders: {value}, {threshold}, {delta}.
# Rendered with actual telemetry values when available; used as static text otherwise.
METRIC_IMPACT_COPY = {
    "avg_memory_utilization": "_Used {value}% of memory — {delta}% above the safe limit._",
    "max_cpu_usage":          "_CPU hit {value}% — {delta}% above the safe threshold._",
    "uptime_days":            "_No restart in {value} days — {delta} days over the recommended limit._",
    "p90_cpu_temp":           "_Running at {value}°C — {delta}°C above the safe limit._",
    "avg_battery_health":     "_Battery at {value}% health — {delta}% below the recommended level._",
    "avg_processor_time":     "_Processor time at {value}% — {delta}% above the threshold._",
}


def _fmt_value(value: float, metric_key: str) -> str:
    if metric_key == "uptime_days":
        return str(int(round(value)))
    if metric_key == "p90_cpu_temp":
        return str(round(value, 1))
    return str(int(round(value)))


def _format_canvas_items(canvas_entries: list, metric_values: dict = None) -> str:
    """Return a numbered mrkdwn list from (canvas_url, label, metric_key) tuples.

    Interpolates {value}, {threshold}, {delta} from metric_values when available.
    """
    items = []
    for i, entry in enumerate(canvas_entries, start=1):
        url, label, metric_key = entry if len(entry) == 3 else (*entry, "")
        link = f"<{url}|{label}>" if url else label
        template = METRIC_IMPACT_COPY.get(metric_key, "")

        impact = ""
        if template:
            raw = (metric_values or {}).get(metric_key)
            threshold = METRIC_THRESHOLDS.get(metric_key)
            if raw is not None and threshold is not None:
                delta = threshold - raw if metric_key in METRIC_LOWER_IS_WORSE else raw - threshold
                try:
                    impact = template.format(
                        value=_fmt_value(raw, metric_key),
                        threshold=_fmt_value(threshold, metric_key),
                        delta=_fmt_value(abs(delta), metric_key),
                    )
                except KeyError:
                    impact = template
            else:
                impact = template

        if impact:
            items.append(f"{i}. {link}\n{impact}")
        else:
            items.append(f"{i}. {link}")
    return "\n\n".join(items)


def build_alert_blocks(
    first_name: str,
    device: str,
    canvas_entries: list,   # list of (canvas_url: Optional[str], label: str, metric_key: str)
    has_replacement: bool = False,
    metric_values: dict = None,
) -> list:
    """
    Build Block Kit blocks for the initial alert DM.

    canvas_entries: list of (canvas_url, label, metric_key) tuples for each flagged metric.
    metric_values: dict of {metric_key: raw_telemetry_value} for threshold interpolation.
    """
    if not canvas_entries:
        return []

    items_text = _format_canvas_items(canvas_entries, metric_values)

    body = (
        f"Hey {first_name} 👋\n\n"
        f"We noticed a few things on your *{device}* that could cause problems if left unaddressed:\n\n"
        f"{items_text}\n\n"
        f"Once you've followed the steps in the canvas, re-run our diagnosis to see if the risk factors are addressed."
    )

    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": body},
        },
        {
            "type": "actions",
            "block_id": "alert_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Re-run diagnosis"},
                    "action_id": "rerun_diagnosis",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Snooze till tomorrow"},
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


def build_still_at_risk_blocks(remaining: dict, canvas_entries: list, metric_values: dict = None) -> list:
    """
    Follow-up shown after re-diagnosis when risk factors remain.

    canvas_entries: list of (canvas_url, label, metric_key) for remaining metrics.
    metric_values: dict of {metric_key: raw_telemetry_value} for threshold interpolation.
    """
    if not canvas_entries:
        return []

    items_text = _format_canvas_items(canvas_entries, metric_values)

    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"Your laptop is still at risk. Please follow these guides:\n\n{items_text}\n\n"
                    f"When you're done, hit *re-run diagnosis* again."
                ),
            },
        },
    ]


def build_article_blocks(article_mrkdwn: str, metric_key: str, is_replacement: bool = False) -> list:
    """Build Block Kit blocks to display a KB article with a resolution button."""
    if is_replacement:
        button_text = "✅ Got it, I'll back up my data"
        action_id = f"done_replacement__{metric_key}"
    else:
        button_text = "✅ Done, re-run the diagnosis"
        action_id = f"done_selfservice__{metric_key}"

    return [
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


def build_snoozed_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "⏰ *Snoozed* — I'll check back with you tomorrow. Take care 😊",
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


def build_opt_out_thanks_blocks() -> list:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Thanks for letting us know — that really helps us improve! 💙",
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
