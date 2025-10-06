---
title: Schema Drift Mitigation Playbook
last_updated: 2025-01-23
tags:
  - data-quality
  - reliability
related:
  - entries/incident-retrospectives.md
  - Docs/research_quick_reference.md
---

## Summary
Recurring schema drift incidents stem from unsignaled upstream changes. This playbook outlines guardrails that reduce customer-facing breakages by 40% based on audit findings.

## Detailed Guidance
1. **Detection Baselines** – Configure automated schema checks on ingestion with drift alerts routed to `#data-quality`. Use the comparison script in the ingestion toolkit.
2. **Contract Reviews** – Require teams to document column-level contracts; store them in the shared glossary. Review quarterly with service owners.
3. **Rollback Process** – Maintain pre-change snapshots and a rollback SOP. Simulate twice per quarter to keep steps current.

## References
- Data Governance Audit (2025-01-23)
- Ingestion Toolkit → `schema_guard.py`

## Follow-up
- Align rollback SOP with the incident command structure by Q2.
