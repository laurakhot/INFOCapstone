"""
Alert sender — called by the telemetry backend to notify a user.

Usage (as library):
    from alerts.sender import send_alert
    send_alert(username="jdoe", device="MBP-JDOE", predictions={"avg_memory_utilization": 0.60})

Usage (as CLI):
    python alerts/sender.py --user=jdoe --device=MBP-JDOE --predictions='{"avg_memory_utilization":0.60}'
"""

import json
import os
import sys

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from alerts.templates import build_alert_blocks, METRIC_LABELS
from rag.articles import is_replacement

load_dotenv()

OPT_OUTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "opt_outs.json")


def _load_opt_outs() -> dict:
    try:
        with open(OPT_OUTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_user(client: WebClient, username: str) -> tuple[str, str]:
    """
    Look up a Slack user by their username (work alias).
    Returns (user_id, first_name).
    Raises ValueError if not found.
    """
    # Paginate through all workspace members
    cursor = None
    while True:
        resp = client.users_list(limit=200, cursor=cursor)
        for member in resp["members"]:
            if member.get("deleted") or member.get("is_bot"):
                continue
            # Match against name (work alias) or display name
            if member.get("name") == username or member.get("profile", {}).get("display_name") == username:
                first_name = (
                    member.get("profile", {}).get("first_name")
                    or member.get("real_name", username).split()[0]
                )
                return member["id"], first_name

        next_cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor

    raise ValueError(f"Slack user not found for username: {username!r}")


def send_alert(username: str, device: str, predictions: dict) -> None:
    """
    Send a proactive laptop health alert DM to the user.

    username:    work alias (same as Slack username)
    device:      device identifier, e.g. "MBP-JDOE-001"
    predictions: dict of metric_key -> confidence float, e.g. {"avg_memory_utilization": 0.60}
    """
    # Check opt-out list first
    opt_outs = _load_opt_outs()
    if username in opt_outs:
        print(f"[send_alert] {username} has opted out — skipping alert.")
        return

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

    # Resolve username → Slack user ID + first name
    user_id, first_name = _get_user(client, username)

    # Open a DM channel
    dm_resp = client.conversations_open(users=[user_id])
    channel_id = dm_resp["channel"]["id"]

    # Build Block Kit blocks
    has_replacement = any(is_replacement(k) for k in predictions if predictions[k] > 0.5)
    blocks = build_alert_blocks(first_name, device, predictions, has_replacement=has_replacement)
    if not blocks:
        print(f"[send_alert] No risk factors above 50% threshold for {username} — skipping.")
        return

    # Fallback text for notifications / accessibility
    risk_labels = [
        METRIC_LABELS.get(k, k)
        for k, v in predictions.items()
        if v > 0.5
    ]
    fallback_text = (
        f"Laptop health alert for {device}: "
        + ", ".join(risk_labels)
    )

    client.chat_postMessage(
        channel=channel_id,
        text=fallback_text,
        blocks=blocks,
    )
    print(f"[send_alert] Alert sent to {username} ({user_id}) for device {device}.")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send a laptop health alert via Slack DM.")
    parser.add_argument("--user", required=True, help="Work alias / Slack username")
    parser.add_argument("--device", required=True, help="Device identifier, e.g. MBP-JDOE-001")
    parser.add_argument(
        "--predictions",
        required=True,
        help='JSON dict of metric_key to confidence, e.g. \'{"avg_memory_utilization":0.60}\'',
    )
    args = parser.parse_args()

    preds = json.loads(args.predictions)
    send_alert(username=args.user, device=args.device, predictions=preds)
