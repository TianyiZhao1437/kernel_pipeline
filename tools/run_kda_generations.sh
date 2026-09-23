#!/bin/bash
# Full KDA generation run: every model in models.yaml, 10 rounds each -- the
# same budget the hca_compress_c128 solutions were generated with, so the two
# tasks' reports compare like with like.
#
# Concurrency is 2, not 4. Generation is API-bound and would parallelise fine,
# but each process evaluates its rounds on the one H200, and the KDA reference
# peaks at 50.16 GiB on the largest workload -- four concurrent processes can
# exceed the card's 140 GiB and OOM one of them mid-run. Two cannot.
#
# The per-round evaluations are only feedback to the model; the number that
# lands in the report comes from tools/run_benchmark.py afterwards, run on its
# own. So the mild timing contention between two concurrent processes does not
# reach any published figure.
set -u
cd /workspace/tianyi/kernel_pipeline

TASK=tasks/kda_prefill_h32_d128
ROUNDS=10
LOGDIR=/tmp/kda_gen
mkdir -p "$LOGDIR"

run_one() {
    local m="$1"
    echo "[$(date +%H:%M:%S)] start $m" >> "$LOGDIR/driver.log"
    python3 -u tools/gen_solution_llm.py \
        --model-name "$m" --task-dir "$TASK" --rounds "$ROUNDS" --stream \
        > "$LOGDIR/$m.log" 2>&1
    echo "[$(date +%H:%M:%S)] done  $m rc=$?" >> "$LOGDIR/driver.log"
}

for pair in "claude-opus-5 glm-5.1" "gpt-6-astra qwen3.8-max"; do
    for m in $pair; do run_one "$m" & done
    wait
done

echo "[$(date +%H:%M:%S)] all models finished" >> "$LOGDIR/driver.log"
