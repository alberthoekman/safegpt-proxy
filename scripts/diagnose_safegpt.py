"""Standalone SafeGPT Message/Execute diagnostic.

Sends one trivial Execute call using the exact same code path as the running proxy
(app.settings.Settings + app.safegpt_client.SafeGPTClient — no FastAPI/router import),
so a result here is directly comparable to what the proxy would see. Useful to:

- Check whether SafeGPT has recovered without running the full Cline/proxy pipeline.
- Produce clean, copy-pasteable evidence for a SafeGPT support ticket.

Usage:
    python scripts/diagnose_safegpt.py
    python scripts/diagnose_safegpt.py --system "You are a test." --prompt "Say pong."
    python scripts/diagnose_safegpt.py --base-url https://api.safegpt.nl --token sk-...
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from app.safegpt_client import SafeGPTClient, SafeGPTStuckExecuteError
from app.settings import Settings

import httpx


def mask_secret(value: str) -> str:
    if not value:
        return "(not set)"
    return f"***{value[-4:]}" if len(value) > 12 else "***"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", default="You are a helpful assistant.", help="systemMessage to send")
    parser.add_argument("--prompt", default="Reply with exactly one word: pong.", help="prompt to send")
    parser.add_argument("--base-url", default=None, help="override SAFEGPT_BASE_URL for this run")
    parser.add_argument("--token", default=None, help="override SAFEGPT_TOKEN for this run")
    parser.add_argument("--model", default=None, help="model to pass through (Execute ignores it, kept for log parity)")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = Settings.load()
    base_url = args.base_url or settings.safegpt_base_url
    token = args.token or settings.safegpt_token
    model = args.model or settings.default_model_id

    print(f"Base URL: {base_url}")
    print(f"Token:    {mask_secret(token)}")
    print(f"Model:    {model}")
    print()

    client = SafeGPTClient(base_url, token)
    try:
        content = await client.execute(system_message=args.system, prompt=args.prompt, model=model)
    except SafeGPTStuckExecuteError as exc:
        print(f"DIAGNOSIS: STUCK UPSTREAM STATE (known bug) — {datetime.now(timezone.utc).isoformat()}")
        print(f"  reported size (SafeGPT): {exc.reported_size} chars")
        print(f"  actual size (this call): {exc.actual_size} bytes")
        print(f"  raw SafeGPT response:    {exc.raw_message[:2000]}")
        print()
        print("  This is the known SafeGPT Message/Execute bug — see claude.md / HANDOFF.md.")
        print("  Not a local sizing issue; do not retry with a smaller prompt.")
        return 1
    except httpx.HTTPError as exc:
        print(f"DIAGNOSIS: UPSTREAM ERROR (not the known stuck-state signature) — {datetime.now(timezone.utc).isoformat()}")
        if isinstance(exc, httpx.HTTPStatusError):
            print(f"  status: {exc.response.status_code}")
            print(f"  body:   {exc.response.text[:2000]}")
        else:
            print(f"  error:  {exc}")
        return 1
    except Exception as exc:
        print(f"DIAGNOSIS: SCRIPT ERROR — {exc!r}")
        return 2
    else:
        print(f"DIAGNOSIS: OK — {datetime.now(timezone.utc).isoformat()}")
        print(f"  content: {content}")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
