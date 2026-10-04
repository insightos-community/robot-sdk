from pathlib import Path

import tomllib


def test_root_is_workspace_not_a_mixed_distribution():
    root = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "project" not in root
    assert root["tool"]["uv"]["workspace"]["members"] == [
        "packages/core",
        "packages/r1pro",
        "packages/franka",
    ]


def test_each_public_package_has_an_independent_distribution():
    projects = {}
    for directory in ("core", "r1pro", "franka"):
        value = tomllib.loads(
            Path(f"packages/{directory}/pyproject.toml").read_text(encoding="utf-8")
        )
        projects[directory] = value["project"]["name"]
    assert projects == {
        "core": "semantic-robot-sdk-core",
        "r1pro": "semantic-robot-sdk-r1pro",
        "franka": "semantic-robot-sdk-franka",
    }
