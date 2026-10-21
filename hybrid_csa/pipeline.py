"""One platform job: prepare -> three training stages -> main -> 32K NIAH, all in SemanticLogKV's torch218.

Training needs nothing beyond what torch218 already has (PyTorch, LitData, PyArrow), so every phase runs with
the evaluation interpreter; there is no separate training environment.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from .config import BASELINE_ROOT, environment_path, load_experiment_config, require_project_file
from .evaluate import (BACKENDS, DEFAULT_CONFIG, DEFAULT_HYBRID_CSA_CONFIG, DEFAULT_EVALUATION_CONFIG,
                       configure_environment, load_hybrid_csa_options, load_eval_suite, validate_output)
from .train import DEFAULT_CONFIG as TRAIN_CONFIG, accumulation_steps, resolve_config


TORCH218_PYTHON = "/home/ma-user/anaconda3/envs/torch218/bin/python"
JOB_CONFIG = BASELINE_ROOT / "configs/hybrid_csa/majob.yaml"
# Variables of the former separate training environment; refused rather than silently ignored.
TRAINING_PYTHON_VARIABLES = ("HYBRID_CSA_TRAIN_PYTHON", "BASELINE_TRAIN_PYTHON")


def first_env(*names, default=None):
    return next((os.environ[name] for name in names if os.environ.get(name)), default)


def configured_output(path=None):
    """The run directory ``output`` of the project's majob.yaml; each run overwrites it."""
    path = require_project_file(path or JOB_CONFIG)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(cfg) != {"output"} or not Path(str(cfg["output"])).is_absolute():
        raise ValueError(f"{path} must set exactly one absolute output")
    return Path(cfg["output"])


def clear_previous_run(output, mode, rank, nodes, keep=()):
    """Overwrite: delete what earlier runs wrote here, so no earlier file can pass for this run's.

    Node 0 deletes the shared result directories; every later write happens inside a torchrun phase that node 0
    joins only afterwards. Each node replaces its own record. An eval-only run keeps train/ (its adapter).
    """
    names = ("eval", "main", "niah") + (() if mode == "eval" else ("train",))
    for kept in keep:
        if any(Path(kept).resolve().is_relative_to((output / name).resolve()) for name in names):
            raise ValueError(f"{kept} lies in a result directory this run overwrites; move it first")
    if rank == 0:
        for name in names:
            path = output / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        for path in output.glob("pipeline_node_*.json"):
            index = path.stem.rsplit("_", 1)[1]
            if not index.isdigit() or int(index) >= nodes:  # records of nodes this run does not have
                path.unlink()
    (output / f"pipeline_node_{rank}.json").unlink(missing_ok=True)


def topology(visible=None):
    nodes = int(first_env("MA_NUM_HOSTS", "NNODES", "GROUP_WORLD_SIZE", default="1"))
    rank = int(first_env("VC_TASK_INDEX", "NODE_RANK", "GROUP_RANK", default="0"))
    configured_gpus = first_env("MA_NUM_GPUS", "NPROC_PER_NODE", "LOCAL_WORLD_SIZE")
    gpus = int(configured_gpus) if configured_gpus else visible
    address = os.environ.get("MASTER_ADDR")
    if not address and os.environ.get("MA_VJ_NAME"):
        job = os.environ["MA_VJ_NAME"]
        address = f"{job}-{os.environ['MA_TASK_NAME']}-{os.environ.get('MA_MASTER_INDEX', '0')}.{job}"
    address = address or "localhost"
    if nodes < 1 or not 0 <= rank < nodes or (gpus is not None and gpus < 1):
        raise ValueError("Invalid node count, node rank or GPU count")
    if visible is not None and gpus > visible:
        raise ValueError(f"Requested {gpus} GPU processes but only {visible} devices are visible")
    if nodes > 1 and (address in ("localhost", "::1") or address.startswith("127.")):
        raise ValueError("Multi-node jobs require a shared reachable MASTER_ADDR")
    port = int(os.environ.get("MASTER_PORT", "6000"))
    ports = {"train": port, "main": int(os.environ.get("EVAL_MASTER_PORT", port + 1)),
             "niah": int(os.environ.get("NIAH_MASTER_PORT", port + 2)),
             "prepare": int(os.environ.get("PREPARE_MASTER_PORT", port + 3))}
    if any(not 1 <= p <= 65535 for p in ports.values()) or len(set(ports.values())) != 4:
        raise ValueError("Prepare/train/main/NIAH ports must be distinct and within 1..65535")
    return dict(nodes=nodes, rank=rank, gpus=gpus, address=address, ports=ports)


def training_gpus(training, topo):
    """Processes per node for training: the most GPUs whose total divides the recipe's global batch.

    The recipe (global batch, steps, token budget) stays fixed; with more GPUs than it can use, training takes
    fewer of them and evaluation still uses all.
    """
    for gpus in range(topo["gpus"], 0, -1):
        world = topo["nodes"] * gpus
        if training["global_batch_size"] % (training["micro_batch_size"] * world) == 0:
            return gpus
    raise ValueError(f"global_batch_size {training['global_batch_size']} with micro_batch_size "
                     f"{training['micro_batch_size']} cannot be split over {topo['nodes']} nodes; "
                     "use a node count that divides it, or change the training YAML and record the new budget")


def distributed_command(python, topo, phase, module, arguments):
    processes = {"prepare": 1, "train": topo.get("train_gpus")}.get(phase) or topo["gpus"]
    command = [python, "-m", "torch.distributed.run", "--nnodes", str(topo["nodes"]),
               "--nproc_per_node", str(processes)]
    if topo["nodes"] == 1:
        command += ["--standalone"]
    else:
        command += ["--node_rank", str(topo["rank"]), "--master_addr", topo["address"],
                    "--master_port", str(topo["ports"][phase])]
    return command + ["-m", module] + arguments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("all", "train", "eval"))
    parser.add_argument("--train-config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--hybrid-csa-config", type=Path, default=DEFAULT_HYBRID_CSA_CONFIG)
    parser.add_argument("--eval-python", help="Interpreter for every phase (default: torch218)")
    parser.add_argument("--backend", choices=BACKENDS, default="hybrid_csa")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--base-checkpoint-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenized-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    # Old adapter/dense/logkv invocations remain evaluation-only.
    args.mode = args.mode or ("eval" if args.adapter or args.backend != "hybrid_csa" else "all")
    if args.mode != "eval" and (args.adapter or args.backend != "hybrid_csa"):
        parser.error("Training requires hybrid_csa without --adapter; use hybrid_csa.train for stage continuation")
    if args.mode == "eval" and args.backend == "hybrid_csa" and not args.adapter:
        parser.error("--mode eval --backend hybrid_csa requires --adapter")
    if args.backend != "hybrid_csa" and args.adapter:
        parser.error("--adapter only applies to hybrid_csa")
    if args.backend == "logkv" and not args.model:
        parser.error("--backend logkv requires --model")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def build_plan(args):
    topo = topology()
    stale = [name for name in TRAINING_PYTHON_VARIABLES if os.environ.get(name)]
    if stale:
        raise ValueError(f"Training runs with the evaluation interpreter (torch218); unset {', '.join(stale)} "
                         "and select the interpreter with --eval-python or HYBRID_CSA_PYTHON")
    python = args.eval_python or first_env("HYBRID_CSA_PYTHON", "BASELINE_EVAL_PYTHON", "BASELINE_PYTHON",
                                          default=TORCH218_PYTHON)
    experiment = require_project_file(args.experiment)
    source = load_experiment_config(experiment)
    if args.base_checkpoint_dir:
        # Keep the Base path convention used by the RULER cache.
        source["ckpt_dir"] = str(args.base_checkpoint_dir.resolve()) + "/"
    base = source["ckpt_dir"]
    if source["context_length"] != 32768:
        raise ValueError("Formal evaluation context_length must be 32768")
    if args.backend == "hybrid_csa" and args.model and args.model.resolve() != Path(base).resolve():
        raise ValueError("HybridCSA must load the same original Base used in training")
    output = args.output or environment_path("HYBRID_CSA_OUTPUT_ROOT", "BASELINE_OUTPUT_ROOT") or configured_output()
    output = Path(output).resolve()
    validate_output(output, {"checkpoint_dir": str(args.model or base), "tokenizer_dir": base})
    settings = load_hybrid_csa_options(args.hybrid_csa_config)
    training = None
    if args.mode != "eval":
        training = resolve_config(args.train_config, {
            "base_checkpoint_dir": base, "output_dir": output / "train",
            "data_dir": args.data_dir.resolve() if args.data_dir else None,
            "tokenized_dir": args.tokenized_dir.resolve() if args.tokenized_dir else None})
        if training["settings"] != settings:
            raise ValueError("Training and evaluation CSA settings differ; align both project YAML files")
        for key in ("data_dir", "tokenized_dir"):
            path = Path(training[key]).resolve()
            if output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError(f"Run output must not overlap {key}: {path}")
    return {"mode": args.mode, "backend": args.backend, "base": base, "model": str(args.model.resolve()) if args.model else None,
            "experiment": str(experiment), "train_config": str(require_project_file(args.train_config)),
            "hybrid_csa_config": str(require_project_file(args.hybrid_csa_config)),
            "python": python, "topology": topo, "training": training,
            "output": str(output), "eval_output": str(output if args.mode == "eval" else output / "eval"),
            "adapter": str(args.adapter.resolve()) if args.adapter else str(output / "train/sparse.pt"),
            "limit": args.limit,
            "cache": {key: os.environ.get(key) for key in
                      ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "RULER_CACHE_DIR", "NLTK_DATA", "TMPDIR")},
            "suites": {suite: load_eval_suite(DEFAULT_EVALUATION_CONFIG, suite)[0].split(",")
                       for suite in ("main", "niah")}}


def execute(plan):
    env = dict(os.environ, HYBRID_CSA_BASE_CHECKPOINT=plan["base"])
    records = {}
    # Fail before preparing/training if the interpreter lacks what a phase imports or CUDA is unusable.
    for phase in (("train", "eval") if plan["mode"] == "all" else (plan["mode"],)):
        command = [plan["python"], "-m", "hybrid_csa.preflight",
                   "--phase", phase, "--base", plan["base"]]
        print(f"[preflight {phase}] {shlex.join(command)}", flush=True)
        records[phase] = json.loads(subprocess.run(command, check=True, capture_output=True, text=True,
                                                  cwd=BASELINE_ROOT, env=env).stdout)
        for name, effect in records[phase].get("missing_optional", {}).items():
            print(f"[preflight {phase}] {name} is not installed: {effect}", flush=True)
    topo = topology(min(record["visible_gpus"] for record in records.values()))
    plan["topology"] = topo
    if plan["training"]:
        cfg = plan["training"]
        topo["train_gpus"] = training_gpus(cfg, topo)
        world = topo["nodes"] * topo["train_gpus"]
        print(f"[train] {topo['train_gpus']} of {topo['gpus']} GPUs per node, {world} processes: global batch "
              f"{cfg['global_batch_size']} = micro batch {cfg['micro_batch_size']} x {world} processes x "
              f"{accumulation_steps(cfg['global_batch_size'], cfg['micro_batch_size'], world)} accumulation steps; "
              f"evaluation uses {topo['gpus']} GPUs per node", flush=True)
    env["LITGPT_EXPECTED_WORLD_SIZE"] = str(topo["nodes"] * topo["gpus"])
    env["NNODES"] = str(topo["nodes"])  # LitData must not infer nodes from all visible GPUs.
    env.pop("GLOBAL_RANK", None)  # torchrun supplies each worker's rank; avoid a stale parent value.
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    keep = [plan["adapter"]] if plan["mode"] == "eval" and plan["backend"] == "hybrid_csa" else []
    clear_previous_run(output, plan["mode"], topo["rank"], topo["nodes"], keep)
    manifest = output / f"pipeline_node_{topo['rank']}.json"
    state = {"plan": plan, "environments": records, "steps": [], "status": "running"}
    # Each node owns a separate manifest on the shared output; "x" refuses a concurrent run of the same node.
    with manifest.open("x") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)

    def run(phase, command, capture=False):
        entry = {"phase": phase, "command": command, "started": datetime.now(timezone.utc).isoformat()}
        state["steps"].append(entry)
        manifest.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print(f"[{phase}] {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=BASELINE_ROOT, env=env, check=True,
                                capture_output=capture, text=True)
        entry["completed"] = datetime.now(timezone.utc).isoformat()
        if capture:
            entry["result"] = json.loads(result.stdout)
        return result

    try:
        if plan["training"]:
            cfg = plan["training"]
            arguments = ["--config", plan["train_config"]]
            for key in ("base_checkpoint_dir", "data_dir", "tokenized_dir", "output_dir"):
                arguments += ["--" + key.replace("_", "-"), cfg[key]]
            run("prepare", distributed_command(plan["python"], topo, "prepare", "hybrid_csa.train",
                                               arguments + ["--prepare-distributed"]))
            run("train", distributed_command(plan["python"], topo, "train", "hybrid_csa.train", arguments))
        if plan["mode"] != "train":
            if plan["backend"] == "hybrid_csa":
                if not Path(plan["adapter"]).is_file():
                    raise FileNotFoundError(f"No completed sparse adapter: {plan['adapter']}")
                run("verify_adapter", [plan["python"], "-m", "hybrid_csa.preflight", "--phase", "eval",
                                       "--base", plan["base"], "--adapter", plan["adapter"],
                                       "--hybrid-csa-config", plan["hybrid_csa_config"]], capture=True)
            arguments = ["--experiment-config", plan["experiment"], "--backend", plan["backend"],
                         "--output", plan["eval_output"], "--hybrid-csa-config", plan["hybrid_csa_config"]]
            if plan["backend"] == "hybrid_csa":
                arguments += ["--adapter", plan["adapter"]]
            if plan["model"]:
                arguments += ["--model", plan["model"]]
            if plan["limit"] is not None:
                arguments += ["--limit", str(plan["limit"])]
            for suite in ("main", "niah"):
                run(suite, distributed_command(plan["python"], topo, suite, "hybrid_csa.evaluate",
                                               arguments + ["--suite", suite]))
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        manifest.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    try:
        configure_environment()
        plan = build_plan(args)
        if args.print_config:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            execute(plan)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode) from error
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
