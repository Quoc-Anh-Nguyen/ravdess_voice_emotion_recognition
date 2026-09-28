# Project Guidelines for Antigravity

## Core Engineering Principles
- **YAGNI & KISS**: Keep solutions simple, minimal, and focused on requirements.
- **Deterministic & Reproducible**: Ensure scripts and ML pipelines run reproducibly.
- **Verification Before Completion**: Run syntax/type checks and tests before declaring work complete.

## Security & Privacy Rules
- Never expose, read, or print sensitive credentials (`.env`, private keys `*.pem`/`*.key`, SSH credentials).
- Sensitive accesses are automatically gated by `.agents/hooks/privacy_guard.py`.

## Repository Structure
- `scripts/`: Data preparation, feature extraction, ML/RCNN/SSL training pipelines.
- `data/`: Dataset metadata and extracted tabular features.
- `figures/` & `reports/`: Evaluation graphs and experimental reports.
- `.agents/skills/`: Custom workflow skills for Antigravity.
- `.agents/hooks/`: Lifecycle hooks for security and quality gates.

