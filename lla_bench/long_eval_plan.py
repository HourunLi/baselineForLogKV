"""The passes of a long-context evaluation over several lengths (eval.sh with eval_lengths / EVAL_LENGTHS).

Each pass uses the niahLogKV task list and length + 1024 generation window.
Native HAttention uses scalar state decay and has no RoPE. The per_length
name is retained for launcher compatibility. No YaRN factor is applied.

EVAL_SUITES chooses the tasks (default all): ruler (NIAH stage: all of RULER, 12 tasks from 256K onward without
ruler_qa_squad, whose 1204 SQuAD paragraphs cannot fill it), babilong, ppl (long_ppl_<L>, LONG_PPL_DATA), books
(long_ppl_books_<L>, LONG_PPL_BOOKS), turns (niah_turns_<L>), longbench2 (longbench2_<L>) and infinitebench
(infinitebench_<L>; INFINITEBENCH_DIR, <HF_HOME>/infinitebench or the hub cache). NIAH_ONLY=1 keeps ruler only. With
the default suites ppl, books and infinitebench are optional: eval.py leaves them out, with the reason, when
their data is not given (or their files are not whole). LongBench v2 must run at every selected length;
missing or incomplete data is an error with a download command. Named suites must run.
InfiniteBench is registered through 256K and runs only at those supported lengths. Asking for it with only
512K/1M selected is an error. RULER uses the documented extended protocols at 512K/1M.

    python long_eval_plan.py <lengths> <suites or ""> <yarn_mode> <niah_only 0/1> <base overrides json or ""> <output dir> [skip]

skip lists what an earlier stage of the same job already measured (majob.sh's main line, eval.sh's full
evaluation): a main task by name, a RULER task at a length as task@length (e.g. niah_single_1@32768); a pass left
with nothing to run is dropped. "python long_eval_plan.py measured <main tasks> <NIAH tasks> <lengths json>" prints
that list for a main line.

prints one line per pass, fields separated by "|": output dir, config overrides (json), RULER lengths (json),
main tasks, NIAH tasks (comma lists, or none), optional tasks.
"""
from __future__ import annotations

import json
import sys

LENGTHS = {"32k": 32768, "64k": 65536, "128k": 131072, "256k": 262144,
           "512k": 524288, "1m": 1048576}
SUITES = ("ruler", "babilong", "ppl", "books", "turns", "longbench2", "infinitebench")
DATA_SUITES = ("ppl", "books", "infinitebench")  # optional with the default suites: their data may be missing
ORIGINAL = 32768  # the trained length, YaRN's original_max_seq_len
MARGIN = 1024
RULER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
         "niah_multivalue", "niah_multiquery", "ruler_vt", "ruler_cwe", "ruler_fwe", "ruler_qa_squad", "ruler_qa_hotpot"]
SQUAD_MAX = 131072  # SQuAD's document pool cannot fill longer contexts
INFINITEBENCH_MAX = 262144  # no registered 512K/1M protocol for these fixed-length source samples


def ruler_tasks(length: int) -> list[str]:
    return [task for task in RULER if length <= SQUAD_MAX or task != "ruler_qa_squad"]


def parse_list(text: str, allowed, what: str) -> list[str]:
    items = [x.strip().lower() for x in str(text).replace(" ", ",").split(",") if x.strip()]
    unknown = [x for x in items if x not in allowed]
    if unknown or not items:
        raise ValueError(f"{what} must be a comma list of {', '.join(allowed)}; got {text!r}")
    return [x for x in allowed if x in items]  # in the canonical order, once each


def tasks(suite: str, name: str) -> list[str]:
    return {
        "babilong": [f"babilong_{name}"],
        "ppl": [f"long_ppl_{name}"],
        "books": [f"long_ppl_books_{name}"],
        "turns": [f"niah_turns_{name}"],
        "longbench2": [f"longbench2_{name}"],
        "infinitebench": [f"infinitebench_{name}"],
    }[suite]


def overrides_for(length: int, base: dict) -> dict:
    # The pass window and task lists are common. Native HAttention has no RoPE.
    out = {**base, "block_size": length + MARGIN}
    out.pop("rope_adjustments", None)
    return out


def plan(lengths: str, suites: str = "", yarn_mode: str = "per_length", niah_only: bool = False,
         base_overrides: dict | None = None, output_dir: str = "", skip=()) -> list[dict]:
    names = parse_list(lengths, tuple(LENGTHS), "EVAL_LENGTHS")
    chosen = parse_list(suites, SUITES, "EVAL_SUITES") if suites else list(SUITES)
    if yarn_mode != "per_length":
        raise ValueError(f"Native HAttention requires per_length passes; got {yarn_mode!r}")
    if niah_only:
        chosen = [s for s in chosen if s == "ruler"] or ["ruler"]
    if suites and "infinitebench" in chosen and all(LENGTHS[n] > INFINITEBENCH_MAX for n in names):
        raise ValueError("InfiniteBench is registered only through 256k; select other suites for 512k/1m")
    passes = []
    for name in names:
        length = LENGTHS[name]
        active = [s for s in chosen if s != "infinitebench" or length <= INFINITEBENCH_MAX]
        main = [t for s in active if s != "ruler" for t in tasks(s, name) if t not in skip]
        niah = [t for t in ruler_tasks(length) if f"{t}@{length}" not in skip] if "ruler" in active else []
        if not main and not niah:
            continue
        optional = [t for s in active if s in DATA_SUITES and not suites for t in tasks(s, name) if t in main]
        folder = f"long{name}"
        passes.append(dict(length=length, name=name, ruler_lengths=[length], main=main, niah=niah,
                           optional=optional, config_overrides=overrides_for(length, base_overrides or {}),
                           output_dir=f"{output_dir.rstrip('/')}/{folder}" if output_dir else folder))
    return passes


def measured(main_tasks: str, niah_tasks: str, lengths: list) -> list[str]:
    """What a main line with these task lists measured, as plan()'s skip: its tasks by name, its NIAH tasks as
    task@length at each of its lengths."""
    done = [t for t in main_tasks.split(",") if t and main_tasks != "none"]
    return done + [f"{t}@{int(n)}" for t in niah_tasks.split(",") if t and niah_tasks != "none" for n in lengths]


def main(argv: list[str]) -> None:
    if argv and argv[0] == "measured":  # measured <main tasks> <NIAH tasks> <lengths json>
        print(",".join(measured(argv[1], argv[2], json.loads(argv[3]))))
        return
    lengths, suites, yarn_mode, niah_only, base, output_dir, *rest = argv
    if not suites and niah_only != "1" and any(n in ("512k", "1m") for n in lengths.lower().split(",")):
        print("[long-eval] InfiniteBench has no registered 512K/1M protocol; omitted at those lengths", file=sys.stderr)
    skip = {t for t in (rest[0] if rest else "").split(",") if t}
    for p in plan(lengths, suites, yarn_mode or "per_length", niah_only == "1",
                  json.loads(base) if base else {}, output_dir, skip):
        print("|".join([p["output_dir"], json.dumps(p["config_overrides"]), json.dumps(p["ruler_lengths"]),
                         ",".join(p["main"]) or "none", ",".join(p["niah"]) or "none", ",".join(p["optional"])]))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}")
