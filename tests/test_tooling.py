"""Local hooks and CI must check code with the same tools."""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def test_pre_commit_runs_the_locked_ruff_version():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = next(p["version"] for p in lock["package"] if p["name"] == "ruff")
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    ruff = next(r for r in config["repos"] if r["repo"].endswith("/ruff-pre-commit"))
    assert ruff["rev"] == f"v{locked}", "update rev in .pre-commit-config.yaml to match uv.lock"
