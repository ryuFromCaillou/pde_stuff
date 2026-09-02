#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import sys
import tempfile
import zipfile
from email.message import EmailMessage
from pathlib import Path


ENV_MAP = {
    "smtp_host": "RESULTS_EMAIL_SMTP_HOST",
    "smtp_port": "RESULTS_EMAIL_SMTP_PORT",
    "username": "RESULTS_EMAIL_USERNAME",
    "password": "RESULTS_EMAIL_PASSWORD",
    "sender": "RESULTS_EMAIL_FROM",
    "recipient": "RESULTS_EMAIL_TO",
    "use_tls": "RESULTS_EMAIL_USE_TLS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a results directory and send it via SMTP email."
    )
    parser.add_argument("--results-dir", required=True, help="Folder to package and send.")
    parser.add_argument("--to", dest="recipient")
    parser.add_argument("--from", dest="sender")
    parser.add_argument("--subject", default="Experiment results")
    parser.add_argument("--note", default="", help="Optional note to prepend to the email body.")
    parser.add_argument("--smtp-host")
    parser.add_argument("--smtp-port", type=int)
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--use-tls")
    return parser.parse_args()


def env_or_value(value: str | int | None, key: str, *, default: str | None = None) -> str | None:
    if value is None or value == "":
        return os.environ.get(ENV_MAP[key], default)
    return str(value)


def parse_bool(value: str | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def require(value: str | None, label: str) -> str:
    if not value:
        raise SystemExit(f"Missing required value for {label}.")
    return value


def summarize_json(summary_path: Path) -> str:
    try:
        payload = json.loads(summary_path.read_text())
    except Exception as exc:
        return f"Failed to parse summary.json: {exc}"

    lines = []
    keys = [
        "dataset",
        "seed",
        "device",
        "final_total_loss",
        "final_data_loss",
        "final_pde_loss",
        "final_tv_loss",
        "data_mse_full_grid",
    ]
    for key in keys:
        if key in payload:
            lines.append(f"{key}: {payload[key]}")

    feature_metrics = payload.get("feature_metrics", {}).get("summary_flat", {})
    for key in sorted(feature_metrics):
        lines.append(f"{key}: {feature_metrics[key]}")

    return "\n".join(lines) if lines else "summary.json present but no recognized keys were found."


def zip_directory(results_dir: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="email-results-"))
    archive_base = temp_dir / results_dir.name
    archive_path = archive_base.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(results_dir))
    return archive_path


def attach_file(message: EmailMessage, path: Path) -> None:
    data = path.read_bytes()
    maintype = "application"
    subtype = "octet-stream"
    if path.suffix.lower() == ".json":
        maintype = "application"
        subtype = "json"
    elif path.suffix.lower() == ".pdf":
        maintype = "application"
        subtype = "pdf"
    elif path.suffix.lower() == ".csv":
        maintype = "text"
        subtype = "csv"
    elif path.suffix.lower() == ".zip":
        maintype = "application"
        subtype = "zip"
    message.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)


def build_message(args: argparse.Namespace, results_dir: Path, archive_path: Path) -> EmailMessage:
    sender = require(env_or_value(args.sender, "sender"), "--from / RESULTS_EMAIL_FROM")
    recipient = require(env_or_value(args.recipient, "recipient"), "--to / RESULTS_EMAIL_TO")

    summary_path = results_dir / "summary.json"
    config_path = results_dir / "config.json"

    body_parts = []
    if args.note:
        body_parts.append(args.note.strip())
    body_parts.append(f"Results directory: {results_dir}")
    if summary_path.exists():
        body_parts.append("Summary metrics:")
        body_parts.append(summarize_json(summary_path))

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = args.subject
    message.set_content("\n\n".join(body_parts))

    if summary_path.exists():
        attach_file(message, summary_path)
    if config_path.exists():
        attach_file(message, config_path)
    attach_file(message, archive_path)
    return message


def send_message(args: argparse.Namespace, message: EmailMessage) -> None:
    smtp_host = require(env_or_value(args.smtp_host, "smtp_host"), "--smtp-host / RESULTS_EMAIL_SMTP_HOST")
    smtp_port = int(env_or_value(args.smtp_port, "smtp_port", default="587"))
    username = require(env_or_value(args.username, "username"), "--username / RESULTS_EMAIL_USERNAME")
    password = require(env_or_value(args.password, "password"), "--password / RESULTS_EMAIL_PASSWORD")
    use_tls = parse_bool(env_or_value(args.use_tls, "use_tls"), default=True)

    if use_tls:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.send_message(message)
        return

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
        server.login(username, password)
        server.send_message(message)


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    archive_path = zip_directory(results_dir)
    try:
        message = build_message(args, results_dir, archive_path)
        send_message(args, message)
    finally:
        try:
            archive_path.unlink()
            archive_path.parent.rmdir()
        except OSError:
            pass

    print(f"Sent results email for {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
