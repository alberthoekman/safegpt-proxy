"""Standalone SafeGPT diagnostic.

Sends one call through the same code path as the running proxy
(app.settings.Settings + app.safegpt_client.SafeGPTClient, no FastAPI/router), so a
result here is directly comparable to what the proxy sees. Useful to check SafeGPT
without running Cline, and to produce copy-pasteable evidence for a support ticket.

Usage (from the project root):
    python -m scripts.diagnose_safegpt
    python -m scripts.diagnose_safegpt --system "You are a test." --prompt "Say pong."
    python -m scripts.diagnose_safegpt --base-url https://api.safegpt.nl --token sk-...
    python -m scripts.diagnose_safegpt --payload logs/failed/<file>.json

--payload replays a request the proxy saved after a SafeGPT error, byte for byte, to the
same URL. Replaying a Message-Create file adds a message to that conversation.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.safegpt_client import SafeGPTClient
from app.settings import Settings


def mask_secret(value: str) -> str:
    if not value:
        return "(not set)"
    return f"***{value[-4:]}" if len(value) > 12 else "***"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="You are a helpful assistant.", help="systemMessage to send")
    parser.add_argument("--prompt", default="Reply with exactly one word: pong.", help="prompt to send")
    parser.add_argument("--payload", default=None, help="replay a saved request from logs/failed/")
    parser.add_argument("--base-url", default=None, help="override SAFEGPT_BASE_URL for this run")
    parser.add_argument("--token", default=None, help="override SAFEGPT_TOKEN for this run")
    return parser.parse_args()


async def send(client: SafeGPTClient, args: argparse.Namespace, model: str) -> str:
    if not args.payload:
        return await client.execute(system_message=args.system, prompt=args.prompt, model=model)
    record = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    url = record["url"]
    accept = "application/json, text/event-stream" if "/Message/Create/" in url else "application/json"
    print(f"Replaying: {url} ({len(json.dumps(record['payload'], ensure_ascii=False))} chars, originally {record.get('status')})")
    # Post directly rather than through SafeGPTClient._post, so a failing replay does not
    # save yet another copy of the same request.
    resp = await client.client.post(url, headers=client.headers(accept), json=record["payload"])
    resp.raise_for_status()
    return resp.text


async def main() -> int:
    args = parse_args()
    settings = Settings.load()
    base_url = args.base_url or settings.safegpt_base_url
    token = args.token or settings.safegpt_token

    print(f"Base URL: {base_url}")
    print(f"Token:    {mask_secret(token)}")
    print()

    client = SafeGPTClient(base_url, token)
    now = datetime.now(timezone.utc).isoformat()
    try:
        content = await send(client, args, settings.default_model_id)
    except httpx.HTTPStatusError as exc:
        print(f"DIAGNOSIS: UPSTREAM ERROR - {now}")
        print(f"  status: {exc.response.status_code}")
        print(f"  body:   {exc.response.text[:2000]}")
        if "string_above_max_length" in exc.response.text:
            print("  SafeGPT measured the prompt after inlining fetched URLs; see HANDOFF.md 'Known SafeGPT quirks'.")
        return 1
    except httpx.HTTPError as exc:
        print(f"DIAGNOSIS: CONNECTION ERROR - {now}")
        print(f"  error:  {exc!r}")
        return 1
    finally:
        await client.close()
    print(f"DIAGNOSIS: OK - {now}")
    print(f"  content: {content[:2000]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
