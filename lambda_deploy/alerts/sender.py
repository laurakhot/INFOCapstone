"""
Alert sender — called directly by lambda_function.py after check_thresholds() identifies problematic users.

send_alert() is called once per user with the username, device, and triggered features from the CSV row.
It looks up the user in Slack, opens a DM, and sends a Block Kit message with canvas links for each flagged metric.

Requires SLACK_BOT_TOKEN to be set as a Lambda environment variable.
"""
import os

from slack_sdk import WebClient

from alerts.templates import build_alert_blocks, METRIC_LABELS, METRIC_BUTTON_LABELS
from rag.articles import get_canvas_url, is_replacement


def _get_user(client: WebClient, username: str) -> tuple:
    """
    Look up a Slack user by their username (work alias) or member ID (U...).
    Returns (user_id, first_name).
    Raises ValueError if not found.
    """
    if username.startswith("U"):
        resp = client.users_info(user=username)
        member = resp["user"]
        first_name = (
            member.get("profile", {}).get("first_name")
            or member.get("real_name", username).split()[0]
        )
        return member["id"], first_name

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

    username: work alias, Slack display name, or member ID (U...)
    device:   device identifier, e.g. "HP EliteBook 8"
    features: dict of {metric_key: raw_value} or list of metric keys
    """
    if not features:
        print(f"[send_alert] No features flagged for {username} — skipping.")
        return

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

    user_id, first_name = _get_user(client, username)

    dm_resp = client.conversations_open(users=[user_id])
    channel_id = dm_resp["channel"]["id"]

    # features may be a dict {metric_key: raw_value} or a plain list of keys
    if isinstance(features, dict):
        metric_values = features
        feature_keys = list(features.keys())
    else:
        metric_values = {}
        feature_keys = features

    has_replacement = any(is_replacement(k) for k in feature_keys)

    canvas_entries = [
        (get_canvas_url(k), METRIC_BUTTON_LABELS.get(k, METRIC_LABELS.get(k, k)), k)
        for k in feature_keys
    ]

    blocks = build_alert_blocks(first_name, device, canvas_entries, has_replacement=has_replacement, metric_values=metric_values)
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
