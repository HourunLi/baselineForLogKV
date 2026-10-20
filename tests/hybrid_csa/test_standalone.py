"""Configuration previews do not depend on torch, other baselines, or a trained adapter."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_imports_stay_inside_private_namespace():
    violations = []
    for path in (ROOT / "hybrid_csa").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
            if any(name.split(".")[0] in {"chunkkv", "litgpt", "utils", "data"} for name in names):
                violations.append(f"{path}:{node.lineno}")
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "sys" and node.attr == "path"):
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("suite", ["main", "niah"])
def test_preview_requires_no_heavy_dependencies_or_adapter_file(suite, tmp_path):
    script = r'''
import runpy
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
class BlockHeavy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "lm_eval", "lightning", "litdata", "litgpt", "chunkkv"}:
            raise AssertionError(f"Unexpected dependency: {fullname}")
sys.meta_path.insert(0, BlockHeavy())
def check_paths(event, args):
    if event == "open" and isinstance(args[0], str) and args[0].endswith((".yaml", ".yml")):
        assert Path(args[0]).resolve().is_relative_to(root), args[0]
sys.addaudithook(check_paths)
sys.argv = ["hybrid_csa.evaluate", "--print-config", "--suite", sys.argv[2], "--adapter", "/not-present.pt"]
runpy.run_module("hybrid_csa.evaluate", run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), suite], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"compress_ratio": 4' in result.stdout and "32768" in result.stdout
    assert '"adapter": "/not-present.pt"' in result.stdout


@pytest.mark.parametrize("flag", ["--experiment-config", "--hybrid-csa-config", "--evaluation-config"])
def test_configs_cannot_reference_other_projects(tmp_path, flag):
    config = tmp_path / "external.yaml"
    config.write_text("{}")
    result = subprocess.run([sys.executable, "-m", "hybrid_csa.evaluate", "--suite", "niah", "--print-config",
                             flag, str(config)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "inside this project" in result.stderr
