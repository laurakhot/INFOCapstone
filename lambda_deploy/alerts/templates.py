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

# Metrics that require hardware replacement (not self-serviceable).
# Must stay in sync with rag/escalation_logic.md.
METRIC_REPLACEMENT = {"avg_battery_health", "avg_processor_time"}

# Format-string templates. Placeholders: {value}, {threshold}, {delta}.
# Rendered with actual telemetry values when available; used as static text otherwise.
# Two-part templates for blockquote rendering.
# "headline" is bold, first line.  "delta" is the second line.
# Placeholders: {value}, {threshold}, {delta}.
METRIC_IMPACT_COPY = {
    "avg_memory_utilization": {
        "headline": "Memory at {value}%",
        "delta":    "{delta}% over the recommended threshold",
    },
    "uptime_days": {
        "headline": "No restart in {value} days",
        "delta":    "{delta} days over the recommended threshold",
    },
    "p90_cpu_temp": {
        "headline": "CPU temperature at {value}°F",
        "delta":    "{delta} degrees over the recommended threshold",
    },
    "max_cpu_usage": {
        "headline": "CPU working {value}% of the time",
        "delta":    "{delta}% over the recommended threshold",
    },
    "avg_processor_time": {
        "headline": "CPU taking around {value} ns to process tasks",
        "delta":    "{delta} ns over the recommended threshold",
    },
    "avg_battery_health": {
        "headline": "Battery at {value}% of its service life",
        "delta":    "{delta}% over the recommended threshold",
    },
}


def _fmt_value(value: float, metric_key: str) -> str:
    if metric_key == "uptime_days":
        return str(int(round(value)))
    if metric_key == "p90_cpu_temp":
        return str(round(value, 1))
    return str(int(round(value)))


def _format_canvas_items(canvas_entries: list, metric_values: dict = None) -> str:
    """Return Slack blockquote blocks from (canvas_url, label, metric_key) tuples.

    Each item renders as:
        > *CPU working 87% of the time*
        > 7% over the recommended threshold
        > <url|Reduce CPU Load (3 min)>
    """
    items = []
    for entry in canvas_entries:
        url, label, metric_key = entry if len(entry) == 3 else (*entry, "")
        link = f"<{url}|{label}>" if url else label
        copy = METRIC_IMPACT_COPY.get(metric_key)

        if copy:
            raw = (metric_values or {}).get(metric_key)
            threshold = METRIC_THRESHOLDS.get(metric_key)
            if raw is not None and threshold is not None:
                delta_val = threshold - raw if metric_key in METRIC_LOWER_IS_WORSE else raw - threshold
                fmt = {
                    "value": _fmt_value(raw, metric_key),
                    "threshold": _fmt_value(threshold, metric_key),
                    "delta": _fmt_value(abs(delta_val), metric_key),
                }
                try:
                    headline = copy["headline"].format(**fmt)
                    delta_line = copy["delta"].format(**fmt)
                except KeyError:
                    headline = copy["headline"]
                    delta_line = copy["delta"]
            else:
                headline = copy["headline"]
                delta_line = copy["delta"]

            items.append(f"> *{headline}*\n> {delta_line}\n> {link}")
        else:
            items.append(f"> {link}")

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

    If any flagged metric is a replacement (not self-serviceable), the action
    buttons switch to Join Queue / Schedule an Appointment / Snooze.
    Otherwise the standard self-service buttons are shown.
    """
    if not canvas_entries:
        return []

    metric_keys = [entry[2] if len(entry) == 3 else "" for entry in canvas_entries]
    is_replacement_alert = any(k in METRIC_REPLACEMENT for k in metric_keys)

    items_text = _format_canvas_items(canvas_entries, metric_values)

    body = (
        f"Hey {first_name} 👋\n\n"
        f"We noticed a few things on your *{device}* that could cause problems if left unaddressed:\n\n"
        f"{items_text}\n\n"
        f"Once you've followed the steps in the canvas, re-run our diagnosis to see if the risk factors are addressed."
    )

    blocks: list = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": body},
        },
    ]

    # ── Primary action buttons ────────────────────────────────────────────
    if is_replacement_alert:
        blocks.append({
            "type": "actions",
            "block_id": "alert_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Join queue"},
                    "action_id": "Join_queue",
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Schedule an appointment"},
                    "action_id": "Schedule_appointment",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Snooze till tomorrow"},
                    "action_id": "Snooze",
                },
            ],
        })
    else:
        blocks.append({
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
            ],
        })

    return blocks


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


