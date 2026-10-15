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


@pytest.mark.parametrize("module,extra_args", [
    ("chunkkv.evaluate", ["--tasks", "niah_single_1"]), ("chunkkv.train", []),
    ("chunkkv.evaluate", ["--suite", "main"]),
    ("chunkkv.evaluate", ["--suite", "longbench"]),
    ("chunkkv.evaluate", ["--suite", "niah"]),
])
def test_print_config_needs_only_local_yaml_and_lightweight_dependencies(module, extra_args, tmp_path):
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
sys.argv = [module, "--print-config", *sys.argv[3:]]
runpy.run_module(module, run_name="__main__")
assert root / "configs/chunkkv/experiment_qwen1.7b_32k.yaml" in yaml_reads
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(PROJECT_ROOT), module, *extra_args],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "32768" in result.stdout
    assert str(PROJECT_ROOT / "outputs") in result.stdout


def test_evaluation_requires_an_explicit_task_selection(tmp_path):
    result = subprocess.run([sys.executable, "-m", "chunkkv.evaluate", "--print-config"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "--suite" in result.stderr


def test_project_contains_no_other_baseline_or_external_checkout_reference():
    """Only ChunkKV code ships here; the removed DMS adapter used to load ../litgpt at runtime."""
    assert not (PROJECT_ROOT / "dms").exists() and not (PROJECT_ROOT / "models").exists()
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if path.suffix in {".py", ".sh", ".yaml", ".toml"} and ".git" not in path.parts and path != Path(__file__):
            text = path.read_text(encoding="utf-8")
            assert "../litgpt" not in text and "litgpt_root" not in text, path


def test_entry_points_reject_configs_outside_the_project(tmp_path):
    from chunkkv.config import load_yaml_chain

    outside = tmp_path / "experiment.yaml"
    outside.write_text("ckpt_dir: /x\ncontext_length: 32768\n")
    result = subprocess.run([sys.executable, "-m", "chunkkv.evaluate", "--suite", "niah", "--print-config",
                             "--experiment-config", str(outside)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "inside this project" in result.stderr
    nested = tmp_path / "nested" / "run.yaml"
    nested.parent.mkdir()
    nested.write_text("config: ../experiment.yaml\n")
    with pytest.raises(ValueError, match="inheritance"):
        load_yaml_chain(nested)
