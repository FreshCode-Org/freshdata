---
title: Growth metrics and traction framework
description: Internal framework for measuring developer adoption, activation, community velocity, and contributor retention.
keywords: open source growth metrics, repository metrics, developer adoption framework
---

# FreshData Growth Metrics & Traction Framework

This document defines the measurement architecture for tracking FreshData's adoption, developer activation, community health, and contributor growth.

> [!IMPORTANT]
> **Measurement Philosophy**:
> GitHub stars are a lagging vanity signal. We prioritize the full developer funnel:
> **Discovery $\to$ Understanding $\to$ Activation $\to$ Trust $\to$ Star $\to$ Contribution $\to$ Advocacy**.
> We never optimize for star counts in isolation.

---

## 1. Funnel Metrics Architecture

```text
Discovery   ──► Unique visitors, search referrals, PyPI views, documentation pageviews
    │
    ▼
Activation  ──► Pip installs (`freshdata-cleaner`), example runs, time-to-first-clean
    │
    ▼
Trust       ──► Audit trail inspection, benchmark reproductions, zero false-repair verification
    │
    ▼
Community   ──► Discussions participation, issue authors, external PRs, time-to-first-review
    │
    ▼
Retention   ──► Repeat contributors, ecosystem integrations, third-party tutorials
```

### 1.1 Discovery Metrics
* **GitHub Repository Traffic**: Unique visitors and total page views (tracked weekly via GitHub Insights).
* **Referral Channels**: Proportion of traffic from search engines, GitHub Explore, Reddit, and Hacker News.
* **Documentation Visitors**: Unique visitors and top pages on `https://freshcode-org.github.io/freshdata/`.
* **PyPI Overview Pageviews**: Visibility of the [`freshdata-cleaner`](https://pypi.org/project/freshdata-cleaner/) package page on PyPI.

### 1.2 Activation Metrics
* **PyPI Package Downloads**: Daily and weekly install counts measured via `pepy.tech` / `pypistats.org`.
* **Installation Accuracy**: Monitoring install error reports to verify users run `pip install freshdata-cleaner` rather than failing on `freshdata`.
* **Example Execution Success**: Zero runtime regressions on all recipes in `examples/`.
* **Time to First Clean**: Target $<3\text{ minutes}$ for a developer to clone, install, and run their first clean.

### 1.3 Community & Contributor Velocity Metrics
* **First-Time Contributors**: Number of unique developers submitting their first PR.
* **Repeat Contributors**: Contributors submitting a second or third contribution within 90 days.
* **Issue & PR Velocity**:
  * First response time on community issues: Target $<24\text{ hours}$.
  * PR review turnaround: Target $<48\text{ hours}$.
* **Discussions Activity**: Active threads in Q&A, Ideas, and Show and Tell.

### 1.4 Growth & Advocacy Metrics
* **Conversion Rate**: Ratio of GitHub stars to qualified unique repository visitors (healthy target: $5\%–8\%$).
* **Forks & Clones**: Active forks actively participating in PR branches.
* **Ecosystem Integrations**: Community-maintained connectors (e.g. Dagster, Prefect, Great Expectations).
* **External Mentions**: Inclusion in curated open-source lists, newsletters (e.g. PyCoders Weekly, Data Elixir), and third-party tutorials.

---

## 2. Internal Traction Targets (30 / 90 / 365 Days)

These figures represent internal planning milestones, not speculative predictions:

| Funnel Area | 30-Day Milestone | 90-Day Milestone | 12-Month Milestone |
|---|:---:|:---:|:---:|
| **GitHub Stars** | 25 – 50 | 100 – 250 | 500 – 1,000+ |
| **Unique Contributors** | 5+ external | 10 – 20 external | 20 – 50 meaningful |
| **Repeat Contributors** | 1 – 2 | 3 – 5 | 10+ active core |
| **Issue / Discussion Interactions** | 10+ meaningful | 30+ threads | Continuous community hub |
| **Weekly PyPI Downloads** | 250+ | 1,000+ | 5,000+ |
| **Ecosystem Integrations** | pandas, Polars, DuckDB | Airflow, Pandera, PyJanitor | Dagster, Prefect, dbt, GX |
| **External Articles & Tutorials** | 1 – 2 | 3 – 5 | 10+ independent blogs |

---

## 3. Review Cadence & Dashboards

* **Weekly Monday Standup**: Maintainers review the past 7 days' PyPI download curve, new external PRs, and unanswered Discussion questions.
* **Monthly Retrospective**: Calculate visitor-to-star conversion, contributor retention rate, and benchmark parity before tagging releases.
