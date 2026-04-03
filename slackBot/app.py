"""
Laptop Health Alert Bot — entry point.

Starts a Slack Bolt app in Socket Mode (no public URL required).

Usage:
    python app.py

Prerequisites:
    1. Copy .env.example to .env and fill in SLACK_BOT_TOKEN + SLACK_APP_TOKEN
    2. pip install -r requirements.txt
    3. Complete Slack App Setup in api.slack.com/apps (see README / plan)
"""

import logging
import os

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from handlers.actions import register_action_handlers
from handlers.events import register_event_handlers

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def create_app() -> App:
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")

    app = App(token=bot_token)

    register_action_handlers(app)
    register_event_handlers(app)

    return app


if __name__ == "__main__":
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if not app_token:
        raise RuntimeError("SLACK_APP_TOKEN is not set. Copy .env.example to .env and fill it in.")

    app = create_app()

    log.info("Starting Laptop Health Alert Bot in Socket Mode...")
    handler = SocketModeHandler(app, app_token)
    handler.start()
