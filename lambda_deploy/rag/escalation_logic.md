# Escalation Routing

Edit the table below to change how the bot routes each metric.

- `type` — `replacement` (hardware issue, cannot self-service) or `self-service`
- `canvas_url` — Slack Canvas link shared directly in the alert DM. Use `-` if not yet available.

| metric_key             | type         | canvas_url |
|------------------------|--------------|------------|
| avg_processor_time     | replacement  | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0AUNA2M940          |
| max_cpu_usage          | self-service | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0ATQKKTMRT          |
| avg_memory_utilization | self-service | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0AUN77TE9E          |
| avg_battery_health     | replacement  | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0ATMJC0UTV          |
| uptime_days            | self-service | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0ASZ3GE087 |
| p90_cpu_temp           | self-service | https://amazonituwcap-wgl1278.slack.com/docs/T0A95N960KS/F0ATQKDUT3P          |