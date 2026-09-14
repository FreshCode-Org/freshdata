"""Contract for the optional TestPyPI dry-run release workflow (#6)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "testpypi.yml"


def _workflow() -> tuple[str, dict]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return text, yaml.safe_load(text)


def test_testpypi_workflow_is_manual_only() -> None:
    _, workflow = _workflow()
    # PyYAML 1.1 treats the unquoted GitHub Actions key `on` as a boolean.
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}


def test_testpypi_publish_uses_trusted_publishing_without_tokens() -> None:
    text, workflow = _workflow()
    publish = workflow["jobs"]["publish-testpypi"]
    assert publish["permissions"]["id-token"] == "write"
    assert publish["environment"]["name"] == "testpypi"
    step = next(s for s in publish["steps"] if "gh-action-pypi-publish" in s.get("uses", ""))
    assert step["with"]["repository-url"] == "https://test.pypi.org/legacy/"
    assert "password" not in step["with"]
    assert "secrets." not in text


def test_testpypi_smoke_installs_from_testpypi_and_prints_version() -> None:
    text, workflow = _workflow()
    assert workflow["jobs"]["smoke"]["needs"] == ["build", "publish-testpypi"]
    assert "--index-url https://test.pypi.org/simple/" in text
    assert 'import freshdata as fd; print(fd.__version__)' in text


def test_release_guide_documents_the_testpypi_verification_command() -> None:
    guide = (ROOT / "RELEASE.md").read_text(encoding="utf-8")
    assert "testpypi.yml" in guide
    assert "pip install --index-url https://test.pypi.org/simple/" in guide
    assert 'python -c "import freshdata as fd; print(fd.__version__)"' in guide
