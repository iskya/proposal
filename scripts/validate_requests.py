#!/usr/bin/env python3
"""Validate one or more request YAML files against the JSON schema."""

import argparse
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
import json


def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_file(path: Path, validator: Draft202012Validator) -> list[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"YAML parse error: {e}"]

    if data is None:
        return ["Empty YAML file"]

    errors = []
    for err in sorted(validator.iter_errors(data), key=lambda e: e.path):
        loc = ".".join([str(p) for p in err.path])
        loc = loc or "<root>"
        errors.append(f"{loc}: {err.message}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Request YAML files")
    ap.add_argument("--schema", default="schemas/request.schema.json")
    args = ap.parse_args()

    schema = load_schema(Path(args.schema))
    validator = Draft202012Validator(schema)

    failed = False
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"❌ File not found: {f}")
            failed = True
            continue

        errs = validate_file(p, validator)
        if errs:
            failed = True
            print(f"❌ {f}")
            for e in errs:
                print(f"   - {e}")
        else:
            print(f"✅ {f}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
