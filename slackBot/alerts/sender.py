"""
Alert sender — called by the telemetry backend Lambda to notify users.

Lambda payload format:
    {
        "problematic_users": [
            {"username": "jdoe", "device": "HP EliteBook 8", "features": ["avg_memory_utilization", "uptime_days"]},
            ...
        ]
    }

Usage (as library):
    from alerts.sender import handle_lambda_payload
    handle_lambda_payload(payload)

Usage (as CLI, single user):
    python alerts/sender.py --user=jdoe --device="HP EliteBook 8" --features='["avg_memory_utilization","uptime_days"]'
"""

import json
import os

from dotenv import load_dotenv
from slack_sdk import WebClient

from alerts.templates import build_alert_blocks, METRIC_LABELS, METRIC_BUTTON_LABELS
from rag.articles import get_canvas_url, is_replacement

load_dotenv()

OPT_OUTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "opt_outs.json")


def _load_opt_outs() -> dict:
    try:
        with open(OPT_OUTS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_user(client: WebClient, username: str) -> tuple:
    """
    Look up a Slack user by their username (work alias).
    Returns (user_id, first_name).
    Raises ValueError if not found.
    """
    cursor = None
    while True:
        resp = client.users_list(limit=200, cursor=cursor)
        for member in resp["members"]:
            if member.get("deleted") or member.get("is_bot"):
                continue
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


def send_alert(username: str, device: str, features: list) -> None:
    """
    Send a proactive laptop health alert DM to the user.

    username: work alias (same as Slack username)
    device:   device identifier, e.g. "HP EliteBook 8"
    features: list of flagged metric keys, e.g. ["avg_memory_utilization", "uptime_days"]
    """
    opt_outs = _load_opt_outs()
    if username in opt_outs:
        print(f"[send_alert] {username} has opted out — skipping alert.")
        return

    if not features:
        print(f"[send_alert] No features flagged for {username} — skipping.")
        return

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

    user_id, first_name = _get_user(client, username)

    dm_resp = client.conversations_open(users=[user_id])
    channel_id = dm_resp["channel"]["id"]

    has_replacement = any(is_replacement(k) for k in features)

    canvas_entries = [
        (get_canvas_url(k), METRIC_BUTTON_LABELS.get(k, METRIC_LABELS.get(k, k)))
        for k in features
    ]

    blocks = build_alert_blocks(first_name, device, canvas_entries, has_replacement=has_replacement)
    if not blocks:
        print(f"[send_alert] No blocks built for {username} — skipping.")
        return

    risk_labels = [METRIC_LABELS.get(k, k) for k in features]
    fallback_text = f"Laptop health alert for {device}: " + ", ".join(risk_labels)

    client.chat_postMessage(
        channel=channel_id,
        text=fallback_text,
        blocks=blocks,
        unfurl_links=True,
    )
    print(f"[send_alert] Alert sent to {username} ({user_id}) for device {device}.")


def handle_lambda_payload(payload: dict) -> None:
    """
    Process the full Lambda JSON payload and send alerts to each flagged user.

    Expected payload shape:
        {"problematic_users": [{"username": str, "device": str, "features": [str, ...]}, ...]}
    """
    for user in payload.get("problematic_users", []):
        try:
            send_alert(
                username=user["username"],
                device=user["device"],
                features=user.get("features", []),
            )
        except Exception as e:
            print(f"[handle_lambda_payload] Failed for {user.get('username')}: {e}")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send a laptop health alert via Slack DM.")
    parser.add_argument("--user", required=True, help="Work alias / Slack username")
    parser.add_argument("--device", required=True, help="Device identifier, e.g. 'HP EliteBook 8'")
    parser.add_argument(
        "--features",
        required=True,
        help='JSON list of flagged metric keys, e.g. \'["avg_memory_utilization","uptime_days"]\'',
    )
    args = parser.parse_args()

    send_alert(username=args.user, device=args.device, features=json.loads(args.features))
