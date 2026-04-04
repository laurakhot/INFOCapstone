# Escalation Routing

Data team: edit the table below to change how the bot routes each metric.

- `article` — only used when platform=shared; specifies the filename in rag/articles/
- `type` — `replacement` or `self-service`
- `platform` — `shared` loads rag/articles/{article}.md
               `yes` loads rag/articles/{mac|windows}/{metric_key}.md

| metric_key             | article               | type         | platform |
|------------------------|-----------------------|--------------|----------|
| cpu_count              | not-self-serviceable  | replacement  | shared   |
| memory_size_gb         | not-self-serviceable  | replacement  | shared   |
| max_cpu_usage          | {metric_key}          | self-service | yes      |
| avg_memory_utilization | {metric_key}          | self-service | yes      |
| uptime_days            | {metric_key}          | self-service | yes      |
| avg_boot_time          | {metric_key}          | self-service | yes      |
| p90_boot_time          | {metric_key}          | self-service | yes      |
