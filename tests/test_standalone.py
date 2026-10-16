"""The project runs without another checkout, and its configuration preview without model dependencies."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_uses_its_own_imports_without_sys_path_injection():
    violations = []
    for path in sorted((REPO_ROOT / "h2okv").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports = [node.module]
            if any(name.split(".")[0] in {"litgpt", "utils", "data", "model"} for name in imports):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: external import")
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "sys" and node.attr == "path"):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: sys.path access")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("extra_args", [
    ["--suite", "main"], ["--suite", "longbench"], ["--suite", "niah"], ["--tasks", "niah_single_1"],
    ["--suite", "niah", "--method", "fullkv"],
])
def test_print_config_needs_only_project_yaml_and_no_model_dependencies(extra_args, tmp_path):
    script = r'''
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

class BlockExternalAndHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"litgpt", "utils", "data", "model", "torch", "lightning", "lm_eval",
                                      "litdata", "datasets", "transformers", "tokenizers"}:
            raise AssertionError(f"Unexpected runtime dependency: {fullname}")

yaml_reads = set()
def check_yaml(event, args):
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if path.suffix in {".yaml", ".yml"}:
            assert path.is_relative_to(root), f"External YAML accessed: {path}"
            yaml_reads.add(path)

sys.meta_path.insert(0, BlockExternalAndHeavyImports())
sys.addaudithook(check_yaml)
sys.argv = ["h2okv.evaluate", "--config", str(root / "configs/qwen3_1.7b_32k_h2o.yaml"), "--print-config",
            *sys.argv[2:]]
runpy.run_module("h2okv.evaluate", run_name="__main__")
assert root / "configs/qwen3_1.7b_32k_h2o.yaml" in yaml_reads
'''
    result = subprocess.run([sys.executable, "-c", script, str(REPO_ROOT), *extra_args],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"max_seq_length": 32768' in result.stdout
    assert str(REPO_ROOT / "outputs") in result.stdout
