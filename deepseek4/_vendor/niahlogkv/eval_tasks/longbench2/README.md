# LongBench v2 task definitions

These task definitions and the prompt template are copied without changes from
[EleutherAI/lm-evaluation-harness, commit d6de81643928d653435c431bae19945d41d32520](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/longbench2).
Their MIT license is included in `LICENSE.md`.

The project's pinned `lm-eval==0.4.9` does not ship these tasks. Keeping the 20
subtasks here makes `longbench2_{32k,64k,128k,256k}` available through the same
repository task manager on both model sizes. The per-length groups keep the
upstream prompt, multiple-choice likelihood scoring and sample-weighted accuracy.

`litgpt/longbench2_patch.py`, the download script and the offline data checker all
read this directory's `dataset_name` values. The 20 subsets contain 503 questions;
each subset still uses the evaluation's configured `limit` when one is supplied.
