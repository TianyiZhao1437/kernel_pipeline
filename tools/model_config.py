#!/usr/bin/env python3
"""Resolve a model name to its gateway, and make the environment agree.

This replaces the four `.env.<model>` files. Each of those exported one base URL
and one API key under seven different variable names -- MODEL_NAME,
OPENAI_BASE_URL/OPENAI_API_KEY (the OpenAI SDK's own), BASE_URL/LLM_API_KEY
(what flashinfer-bench's kernel_generator reads),
OPENAI_COMPAT_BASE_URL/OPENAI_COMPAT_API_KEY (namespace-happy clients) -- with
nothing in the repository recording which alias any given consumer actually
read. That is how a model looks configured and still fails.

`models.yaml` records each model once. This module reads it and then *still*
exports all seven spellings, because the vendored `KernelGenerator`
(third_party/flashinfer-bench/examples/kernel_generator/kernel_generator.py:53)
does `os.getenv("LLM_API_KEY")` and is not ours to edit. The alias layer is now
a shim over one source of truth rather than the source itself.

Usage as a library:

    from model_config import load_models, resolve, apply
    cfg = resolve("claude-opus-5")      # Config(model=, base_url=, api_key=, author=)
    apply(cfg)                           # exports the seven names

Usage as a preflight:

    python3 tools/model_config.py --list
    python3 tools/model_config.py --check claude-opus-5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO / "models.yaml"
EXAMPLE_REGISTRY = REPO / "models.example.yaml"

# Every spelling the consumers use, mapped to the canonical field it carries.
# Order matters only for `--list` output; all three get the same value.
_BASE_URL_ALIASES = ("OPENAI_BASE_URL", "BASE_URL", "OPENAI_COMPAT_BASE_URL")
_API_KEY_ALIASES = ("OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_COMPAT_API_KEY")
_MODEL_ALIASES = ("MODEL_NAME", "MODEL")


class ConfigError(RuntimeError):
    """A model could not be resolved, or the registry is malformed."""


@dataclass
class Config:
    name: str
    model: str
    base_url: str
    api_key: Optional[str] = None
    author: Optional[str] = None

    @property
    def solution_author(self) -> str:
        """The `solutions/<author>/` directory this model lands under.

        Defaults to the registry key with `.` and `/` replaced, which is what
        the existing trace set already uses: `qwen3.8-max` -> `qwen3_8_max`,
        `openai/gpt-6-astra` -> `gpt_6_astra`. Stated here rather than inferred
        at each call site so a rename cannot silently split one model's
        solutions across two directories.
        """
        if self.author:
            return self.author
        return self.name.replace(".", "_").replace("/", "_")


def _registry_path(path: Optional[pathlib.Path] = None) -> pathlib.Path:
    if path is not None:
        return path
    if DEFAULT_REGISTRY.is_file():
        return DEFAULT_REGISTRY
    if EXAMPLE_REGISTRY.is_file():
        # Not an error on its own -- `--list` against the template is a useful
        # thing to do -- but resolve() refuses below, because every key in it
        # is a placeholder.
        return EXAMPLE_REGISTRY
    raise ConfigError(
        f"neither {DEFAULT_REGISTRY.name} nor {EXAMPLE_REGISTRY.name} exists. "
        f"Copy {EXAMPLE_REGISTRY.name} to {DEFAULT_REGISTRY.name} and fill it in."
    )


def load_models(path: Optional[pathlib.Path] = None) -> Dict[str, dict]:
    """Read the raw `models:` mapping. Does not validate or resolve keys."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyyaml is a declared dep
        raise ConfigError(f"pyyaml is required to read the registry: {exc}") from exc

    src = _registry_path(path)
    if not src.is_file():
        raise ConfigError(f"registry {src} does not exist")
    data = yaml.safe_load(src.read_text()) or {}
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError(f"{src} has no non-empty `models:` mapping")
    return models


def resolve(name: str, path: Optional[pathlib.Path] = None) -> Config:
    """Resolve `name` to a Config.

    A name in the registry wins. Otherwise `name` is treated as a bare model
    string and the endpoint and key come from the environment, which keeps the
    "point it at a gateway and try" path working without an edit to the
    registry. The API key is read from the environment either way, so a key
    never has to be written into a file at all if the environment already has
    one; the registry entry may name the variable via `api_key_env`.
    """
    models = load_models(path)

    if name not in models:
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        if not base_url:
            known = ", ".join(sorted(models))
            raise ConfigError(
                f"{name!r} is not in the registry ({known}) and no "
                "OPENAI_BASE_URL/BASE_URL is set, so it cannot be treated as a "
                "bare model string either."
            )
        return Config(
            name=name,
            model=name,
            base_url=base_url.rstrip("/"),
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        )

    entry = models[name] or {}
    model = entry.get("model")
    base_url = entry.get("base_url")
    if not model:
        raise ConfigError(f"model {name!r} in the registry has no `model:` string")
    if not base_url:
        raise ConfigError(f"model {name!r} in the registry has no `base_url:`")

    api_key = entry.get("api_key")
    if not api_key and entry.get("api_key_env"):
        api_key = os.environ.get(entry["api_key_env"])
    if not api_key:
        # Last resort: an ambient key. Useful when the registry is the template
        # and the caller exported a key instead of copying the file.
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")

    return Config(
        name=name,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        author=entry.get("author"),
    )


def apply(cfg: Config) -> None:
    """Export every alias the consumers read, from this one resolved Config."""
    for key in _MODEL_ALIASES:
        os.environ[key] = cfg.model
    for key in _BASE_URL_ALIASES:
        os.environ[key] = cfg.base_url
    if cfg.api_key:
        for key in _API_KEY_ALIASES:
            os.environ[key] = cfg.api_key


def describe(cfg: Config) -> str:
    """One line, never including the key itself."""
    key = "set" if cfg.api_key else "MISSING"
    return f"{cfg.name:<18} model={cfg.model:<28} base_url={cfg.base_url:<52} key={key}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", type=pathlib.Path, default=None,
                    help=f"registry file (default: {DEFAULT_REGISTRY.name})")
    ap.add_argument("--list", action="store_true", help="list registered models and exit")
    ap.add_argument("--check", metavar="NAME", action="append", default=None,
                    help="resolve one model and report whether it is usable")
    ap.add_argument("--export", metavar="NAME", default=None,
                    help="print shell exports for one model, with the key redacted")
    args = ap.parse_args()

    try:
        models = load_models(args.registry)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.list or (not args.check and not args.export):
        print(f"registry: {_registry_path(args.registry)}  ({len(models)} models)")
        for name in sorted(models):
            try:
                print("  " + describe(resolve(name, args.registry)))
            except ConfigError as exc:
                print(f"  {name:<18} INVALID: {exc}")
        return 0

    status = 0
    for name in args.check or []:
        try:
            cfg = resolve(name, args.registry)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
            continue
        print(describe(cfg))
        if not cfg.api_key:
            print(f"  warning: no API key resolved for {name!r}", file=sys.stderr)
            status = 1

    if args.export:
        cfg = resolve(args.export, args.registry)
        for key in _MODEL_ALIASES:
            print(f"export {key}={cfg.model}")
        for key in _BASE_URL_ALIASES:
            print(f"export {key}={cfg.base_url}")
        for key in _API_KEY_ALIASES:
            print(f"export {key}=<resolved at runtime>")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
