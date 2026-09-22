#!/usr/bin/env python3
"""Normalise the .env.<model> gateway files onto one variable naming scheme.

All three gateways speak the OpenAI-compatible protocol, but the files had
drifted apart: one used OPENAI_COMPAT_BASE_URL, the others OPENAI_BASE_URL, and
flashinfer-bench's own kernel_generator example reads LLM_API_KEY / BASE_URL. A
script rather than hand-editing because the values are live API keys -- this
reads and rewrites them without ever printing one.

Canonical names, plus aliases so a tool expecting any of the common spellings
works unchanged:

    OPENAI_BASE_URL / OPENAI_API_KEY   the OpenAI SDK's own variables
    MODEL_NAME                          which model this file selects
    BASE_URL / LLM_API_KEY              what examples/kernel_generator reads
    OPENAI_COMPAT_BASE_URL / _API_KEY   kept; some gateways' clients use these

Usage:  python3 tools/normalise_env_files.py [--check]
"""

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Any of these source keys may carry the value; first match wins.
BASE_URL_KEYS = ("OPENAI_BASE_URL", "OPENAI_COMPAT_BASE_URL", "BASE_URL")
API_KEY_KEYS = ("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY", "LLM_API_KEY")
MODEL_KEYS = ("MODEL_NAME", "MODEL")

TEMPLATE = """\
# OpenAI-compatible gateway for {model}.
# Load with:  set -a; . ./{filename}; set +a
#
# The same value is exported under several names on purpose: different tools
# read different variables, and this file is the single place the credential
# lives. See tools/normalise_env_files.py.

export MODEL_NAME={model}

# OpenAI SDK (openai.OpenAI() picks these up with no arguments)
export OPENAI_BASE_URL={base_url}
export OPENAI_API_KEY={api_key}

# flashinfer-bench examples/kernel_generator
export BASE_URL={base_url}
export LLM_API_KEY={api_key}

# Clients that namespace the OpenAI-compatible endpoint
export OPENAI_COMPAT_BASE_URL={base_url}
export OPENAI_COMPAT_API_KEY={api_key}
"""


def parse(path: pathlib.Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key] = val
    return values


def first(values: dict, keys: tuple) -> str | None:
    for k in keys:
        if values.get(k):
            return values[k]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    paths = sorted(REPO.glob(".env.*"))
    paths = [p for p in paths if p.suffix != ".bak"]
    if not paths:
        print("no .env.* files found")
        return 1

    failures = 0
    for path in paths:
        values = parse(path)
        base_url = first(values, BASE_URL_KEYS)
        api_key = first(values, API_KEY_KEYS)
        model = first(values, MODEL_KEYS) or path.name.removeprefix(".env.")

        missing = [n for n, v in (("base_url", base_url), ("api_key", api_key)) if not v]
        if missing:
            print(f"  FAIL  {path.name}: no value for {', '.join(missing)}")
            failures += 1
            continue

        body = TEMPLATE.format(
            model=model, base_url=base_url, api_key=api_key, filename=path.name
        )
        if args.check:
            state = "up to date" if path.read_text() == body else "would be rewritten"
            print(f"  {path.name}: {state}")
        else:
            path.write_text(body)
            # Credentials: owner-only. They were world-readable before.
            path.chmod(0o600)
            print(f"  ok    {path.name}  model={model}  key len={len(api_key)}  mode=600")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
