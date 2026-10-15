"""ChunkKV configuration must work without another checkout or model dependencies."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_chunkkv_uses_package_imports_without_sys_path_injection():
    violations = []
    for path in sorted((PROJECT_ROOT / "chunkkv").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports = [node.module]
            if any(name.split(".")[0] in {"litgpt", "utils", "data"} for name in imports):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: external import")
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "sys" and node.attr == "path"):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: sys.path access")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("module", ["chunkkv.evaluate", "chunkkv.train"])
def test_print_config_needs_only_local_yaml_and_lightweight_dependencies(module, tmp_path):
    script = r'''
from pathlib import Path
import runpy
import sys

root, module = Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path.insert(0, str(root))

class BlockExternalAndHeavyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"litgpt", "utils", "data", "torch", "lightning", "lm_eval", "litdata"}:
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
sys.argv = [module, "--print-config"]
runpy.run_module(module, run_name="__main__")
assert root / "configs/chunkkv/experiment_qwen1.7b_32k.yaml" in yaml_reads
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT), module],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "32768" in result.stdout
    assert str(PROJECT_ROOT / "outputs") in result.stdout
