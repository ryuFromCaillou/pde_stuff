---
name: email-results-mailer
description: Package experiment outputs such as run_results folders, attach a zip and selected summary files, and send them through SMTP email. Use this skill when the user wants Codex to email training results, metrics, plots, or artifacts from the local workspace.
---

# Email Results Mailer

## Overview

Use this skill when a user wants local experiment results emailed from the current machine. The skill packages a target results folder, extracts useful summary text from `summary.json` when present, and sends an email with attachments through a configured SMTP account.

## When To Use

- The user asks to email `run_results/...` outputs.
- The user wants the latest experiment summary delivered with attachments.
- The user wants a repeatable "train, package, and send" workflow instead of manual email composition.

Do not use this skill if SMTP credentials are unavailable or if the user has not authorized sending email from the machine.

## Workflow

1. Identify the folder or files to send.
2. Prefer a results folder containing `summary.json`, `config.json`, plots, or model artifacts.
3. Run `scripts/send_results_email.py` with either explicit SMTP flags or environment variables.
4. Include a short subject and optional note from the user.
5. Report exactly what was attached and whether the send succeeded.

## SMTP Inputs

The script accepts either CLI flags or environment variables:

- `RESULTS_EMAIL_SMTP_HOST`
- `RESULTS_EMAIL_SMTP_PORT`
- `RESULTS_EMAIL_USERNAME`
- `RESULTS_EMAIL_PASSWORD`
- `RESULTS_EMAIL_FROM`
- `RESULTS_EMAIL_TO`
- `RESULTS_EMAIL_USE_TLS`

Command-line flags override environment variables.

## Preferred Invocation

Use:

```bash
source .venv/bin/activate && python skills/email-results-mailer/scripts/send_results_email.py \
  --results-dir run_results/eql_joint_training/allen_cahn_generated/seed_000 \
  --to you@example.com \
  --subject "PDE training results"
```

If the folder contains `summary.json`, the script includes a compact metrics section in the email body automatically.

## Notes

- The script creates a zip archive in a temporary directory and cleans it up after sending.
- By default it attaches `summary.json` and `config.json` directly when present, plus a zip of the full results directory.
- If the environment blocks network access, request approval before retrying the send outside the sandbox.
