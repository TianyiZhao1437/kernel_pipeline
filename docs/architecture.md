# kernel_pipeline — architecture

## 0. Why this document, and what it is not

The repository already *performs* this pipeline for one task. `tools/` has eleven
scripts that stage, validate, benchmark and score `hca_compress_c128`, and the
record in `tasks/HCA.md` shows what each of them caught. What is missing is not a
capability but a **shape**: the steps live in prose and shell history, every path
is a hardcoded module constant, and nothing connects `tools/gen_solution_llm.py`
to `tools/run_benchmark.py` except a human remembering the order.

So this document is written to be **wrong in a checkable way where it disagrees
with the repository**, not to be aspirational. Every section names the existing
tool it formalises, the constant it replaces, or the gap it creates. Where the
current code already proves a convention is right — the baseline author default,
the two-root split, the random-input impossibility — that proof is cited rather
than re-argued.

The thing being formalised is a **two-stage pipeline over a self-describing
artifact**:

```
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │  1  TASK GENERATION      │          │  2  TRACE RUNNING        │
  │                          │          │                          │
  │  1.1 idea intake ────────┼──┐       │  2.1 select models       │
  │  1.2 source resolution   │  │       │  2.2 verify env          │
  │  1.3 task assembly       │  │       │  2.3 generate + run      │
  │  1.4 input materialise   │  │       │  2.4 report              │
  │  1.5 execute + gate      │  │       │                          │
  └──────────────────────────┘  │       └──────────────────────────┘
                                │                    ▲
                       writes   ▼                    │  reads
                            ┌─────────────────────────────────┐
                            │  task/ (definition, workloads,  │
                            │  solutions, eval_config,        │
                            │  task.yaml, PROVENANCE)         │
                            └─────────────────────────────────┘
```

The two stages **share no Python**. They meet at a directory. Everything stage 1
promises, stage 2 must be able to re-derive from files alone, because stage 2's
job is to be able to say "this task is not runnable, and here is which promise
failed" without reading stage 1's logs.

## 1. The design principle this whole pipeline exists to enforce

`tasks/HCA.md` §2 is titled *"Four schema constraints that are not in the docs"*
and every item in it was **discovered by running the real parser, after a draft
that looked entirely plausible had already violated it**. The same pattern recurs
in §7.7: random inputs made two of five Definition inputs physically impossible,
and that was invisible until a real model was probed.

The generalisable rule, and the reason for the verifier gate rather than a
checklist:

> **A claim about the task is not part of the task until it has been executed.**
> Prose, a paper, a plausible-looking draft, and a model's own description are
> all *leads*. They become facts only by (a) being pinned to a revision and
> (b) surviving a check that would have caught the counterexample.

Concretely this is why `validate_task.py` exists next to `validate_dataset.py`:
schema validity and *semantic* validity are different properties, and the second
one has repeatedly been the one that failed.

## 2. Stage 1 — task generation

### 2.1 Step 1: idea intake and source resolution

Input: a kernel name (e.g. `hca_compress`). Output: a **source pack**, not a
task.

The temptation is to treat this as "search and read". The failure mode is that
search returns a *consensus description* that is confidently wrong — exactly the
"four constraints not in the docs" trap. So the step is specified as producing
four kinds of evidence, each with an explicit trust level:

| Evidence | What it is for | Trust |
|---|---|---|
| **Executable source** | a pinned implementation to diff against | highest — it runs |
| **Model weights + config** | exact tensors, shapes, dtype, eps, rope theta | high — hashable |
| **Math / formula** | the semantics to implement | medium — must be reconciled against the source |
| **Torch description** | the op-level expression, what a `torch.compile` baseline would fuse | medium — measured, not read |

The deliverable is a **provenance record** (one per task) naming, for each
Definition input, where it came from and whether that source is *exact* or
*synthesised*. `tools/model_probe.py`'s docstring is the worked example of the
table this belongs in: four of five HCA inputs are **exact** (`norm.weight`,
`DeepseekV4RotaryEmbedding`, `wgate`+`ape`, `wkv`), and exactly one — the hidden
state — is not. That split was expensive to discover and must not be re-derived
by hand per task.

Two rules the step must enforce:

1. **Revision pinning is mandatory, and it is a security property, not
   tidiness.** `model_probe.py` pins `MODEL_REVISION` because "a hub repo can be
   force-pushed or re-quantised in place". Every measurent in this repository
   derives from those bytes; an unpinned fetch makes the numbers
   non-reproducible *and* makes the task a supply-chain dependency on a mutable
   third-party artifact.
2. **Search output is never transcribed into the reference.** The reference is
   written against the executable source, then *reconciled* with the math.
   `HCA.md` §8 item 9 is precisely this reconciliation left open: the reference
   was written from vLLM and has not been diffed line-by-line against the
   native `DeepseekV4HCACompressor`. That item is a **gate** in this design, not
   a nice-to-have.

**Hardware detection** belongs here, and only here. It sets the *default* target
for everything downstream — the plan's "detect local GPU as default hardware" —
and the detection has a hard constraint attached: per `CLAUDE.md` §12, the
driver is host-injected and the wheel constraint is set by **compute
capability**, not by driver version. A task whose reference needs a kernel the
local card cannot run is not a task this instance can gate. Record the detected
target in `task.yaml`; treat a mismatch at stage 2 as a hard error, not a
warning.

### 2.2 Step 2: task assembly, and the four contracts

Output: a flat task directory in the layout `stage_trace_set.py` already expects
— Definition, Solutions, workload sweep, `eval_config.yaml`, all in one place so
it "reads as a unit and diffs as a unit".

Components, with the existing tool that builds each:

- **Definition** — `tools/build_hca_task.py` is the only bespoke builder and is
  task-specific; the reusable part is the schema, so this step is a
  generalisation of that script, not a new format.
- **Reference** — the executable implementation. Must take exactly
  `len(inputs)` positional arguments (`validate_task.py:check_arity`).
- **Baseline** — the incumbent a candidate must beat. `build_solution.py`'s
  default author is `tim.zhao` and the comment there records *why*: a seed was
  once mislabelled `claude-opus-5` purely because a default said so, "which made
  a hand-written baseline read as a model attempt". The rule is therefore
  **author = who actually wrote it**, and the plan's "baseline author defaults to
  tim.zhao" is correct only in that narrow sense — it is a default for
  hand-authored solutions, and `tools/gen_solution_llm.py` must keep passing
  `--author` explicitly for generated ones. Conflating the two is the bug that
  comment was written to prevent.
- **Workloads** — the sweep. Two axes of care: *coverage* (the plan should say
  which regimes must be represented, and `HCA.md` §7.6 shows why — the sweep was
  truncated to the middle of the range and the top of the range is where the two
  shipped solutions diverge by 3.3x) and *the `speedup_factor` floor* (§7.5:
  small workloads measure op-dispatch overhead, and their `speedup_factor` "must
  not feed a ranking").
- **`eval_config.yaml`** — tolerances and per-definition config. This is
  **required, not optional**, and `run_benchmark.py`'s docstring records the
  reason: `BenchmarkConfig.default()` bundles `required_matched_ratio: 1.0`,
  i.e. bitwise equality, which no kernel over a 128-token softmax can satisfy.
  Omitting the config does not produce a lenient run; it produces a *wrong*
  verdict.

The **four contracts** are catalogued by name in
[`contracts.md`](contracts.md#the-contract-catalogue), which is the authority for
what each one asserts and how it is checked. The short version, and the mapping
to the plan's four groups:

| # | Contract | Asserts |
|---|---|---|
| C1 | **Schema** | parses under the real pydantic models; shapes resolve; arity matches |
| C2 | **Semantics** | it computes the stated op — no shortcut that reports the right number without doing the work |
| C3 | **Inputs** | every workload input is physically realisable for this op |
| C4 | **Trace-ready** | reproducible byte-for-byte from pinned sources on this target |

C2 and C3 are the ones the current tooling does **not** cover, and they are the
two the plan is right to insist on ("无 hack，合法，符合实际工程语义").

### 2.3 Step 3: input materialisation

The plan's step 3 — "if resources allow, generate directly; otherwise pull a
minimal runnable model and actually measure" — is the right shape, and
`tools/gen_workload_blobs.py` is already the right implementation. It should be
promoted from "a script for HCA" to the pipeline's answer for any task whose
inputs are physically constrained.

The design decision worth writing down is **when a real model is needed at all**,
because that is what the plan means by "if resources allow" and the naive
reading (always run the model) is what makes the fallback look like a
compromise. The repository already contains the criterion:

> A task needs real inputs when a *statistic of the input distribution* changes
> the kernel's behaviour or the verifier's verdict. Not when it merely changes
> the numbers.

HCA is a yes on every count: `rms_norm_weight` as `N(0,1)` is half negative with
a median near zero against a real tensor that is 503/512 positive and spans 190x;
`cos_cache`/`sin_cache` as `N(0,1)` are not a rotation at all (real entries
satisfy `cos² + sin² = 1`); and the UE8M0 exponent — the entire subject of the
quantiser — landed in `[-8, -4]` of a representable `[-22, 120]` under random
inputs, i.e. the evaluated range was a corner of the real one.

There is a **third option** between "run the model" and "synthesise from
scratch", and the repo is already using it: probe the model for the small tensors
that are exact (`model_probe.compressor_weights`, `rope_tables`) and synthesise
only the activation, lifting real per-channel marginals from a *cheaper* model
(`v2lite_activations`) rather than inventing them. That is what made HCA's inputs
faithful at 16B scale instead of 148 GiB. It should be the documented default,
with full-model execution as the escalation.

Whatever the source, it is **pinned** (`MODEL_REVISION`) and the blobs are
**hash-listed** (`tasks/hca_compress_c128/blobs.sha256`), because stage 2's
numbers are only meaningful relative to the exact bytes stage 1 froze.

### 2.4 Step 4: first execution and gate

Run the reference and the baseline. This is `tools/run_benchmark.py` over a
root staged by `tools/stage_trace_set.py`, and it produces the first Traces —
which is also the first *evidence*, because `HCA.md` §7.5 is the record of the
first trace overturning a premise (the seed solution was 19x off roofline by
design; see the memory note on why that is deliberate).

The gate's real content is Step 5, so they are specified together.

### 2.5 Step 5: the gate

This is the step the plan describes as "保证生成的task完整，无hack，合法，符合实际
工程语义，且ready for trace", and it must be split, because half of it is
mechanisable and half of it is not.

**Mechanisable — becomes `tools/verify_task.py`.** Every item is an existing
check or a one-line generalisation of one:

1. `validate_task.py` — schema, arity, input/output shape resolution (via
   `fib_shim`, which is how the parsers load without the package `__init__`).
2. `validate_dataset.py` — the vendored validator over a *staged* root. Note the
   trap it documents: the validator discovers the dataset from path depth and
   **silently skips** wrong-depth files, so a hand-staged root reports
   "0 definitions" rather than an error. Staging must go through
   `stage_trace_set.py`.
3. **Anti-hack checks** (C2), which currently do not exist. The cheap, high-value
   ones: outputs must be *derived* from inputs (not aliased to a constant, not
   read from the reference); the reference must not call the baseline or the
   candidate; a solution's entry must not branch on a workload-identity signal.
4. **Input-realism checks** (C3): for each declared input, is it
   `SafetensorsInput` with a hash, or `RandomInput`? If random, the task must
   *declare* that it is distribution-insensitive, and that declaration is what a
   reviewer signs off on — it is not an assertion the verifier can make alone.
5. **Blob integrity** — `blobs.sha256` verifies.
6. **Runs at all** — reference builds, matches itself at the required ratio, and
   the baseline traces. `validate_dataset.py`'s `benchmark` check does this, but
   note `HCA.md` §8 item 8: that check carries a **hardcoded `BenchmarkConfig`**,
   so its verdict must be reconciled against the task's own `eval_config.yaml`
   or it can disagree with the real run.

**Not mechanisable — stays a human sign-off, and the pipeline should say so
rather than pretend.** "符合实际工程语义" is a judgement. What the pipeline owes
the reviewer is a *narrow* question with the evidence attached: the provenance
record (which claims are exact vs synthesised), the open reconciliation items
(HCA's §8 item 9), and the coverage argument for the sweep. A gate that reports
"13/13 checks passed" while the semantic question is unanswered is worse than one
that prints "12 automated checks passed; 1 semantic review outstanding".

**The gate blocks stage 2.** A task that has not passed is not "a task with
warnings"; stage 2 must refuse it, or the failure surfaces as an unattributable
bad trace.

## 3. Stage 2 — trace running

### 3.1 Step 1: model selection

Input: a task that passed the gate, plus a set of models. Output: a run
manifest.

The thing that must be specified here is the **model configuration contract**,
because the current mechanism is a loose convention. There are four `.env.*`
files, each defining **seven aliases for two values**:

```
MODEL_NAME  OPENAI_BASE_URL  OPENAI_API_KEY          (the OpenAI trio)
BASE_URL    LLM_API_KEY                              (a second pair)
OPENAI_COMPAT_BASE_URL  OPENAI_COMPAT_API_KEY        (a third)
```

Nothing in the repo says which set `gen_solution_llm.py` reads, which is how a
model can appear configured and still fail. The contract should be: one
`models.yaml` naming, per model, the env file, the model string, and a
**capability declaration** (context length, whether it emits tool calls, whether
it streams reasoning). The run manifest is then a list of model names plus the
task, and it is the *only* input to step 3.

### 3.2 Step 2: environment verification

The plan says "检查模型env真实可用". This must be **active, not configured** —
the distinction is the whole point of the step. A `.env` file existing, or a key
parsing, proves nothing. The failure mode this repo has actually hit is a
gateway returning **no visible content**: `claude-opus-5` and `gpt-6-astra` each
dropped several replies mid-run (4 and 8 respectively, each 1–2 chunks in ~0.2s).
Those calls **still bill** — and per the memory note, the crash lands *inside*
the call, so a retry that wraps only the HTTP request does not catch it.

So verification is: a one-call smoke test per model that asserts **non-empty
content**, not HTTP 200. It costs one cheap call and it prevents starting a
10-round generation against a dead endpoint.

### 3.3 Step 3: generation and trace

One trace per model — the plan's "每个模型跑一次trace".

Two invariants, both already learned the hard way:

1. **The generation root and the measurement root must be the same root.**
   `gen_solution_llm.py` comments at length on this: it was pinned at the old
   20-workload `hca_c128_v4`, and if generation evaluates against a smaller sweep
   than the trace is later judged on, "the round-by-round feedback the model
   optimises against is a different — and smaller — sweep than the trace it is
   finally judged on". Today both defaults point at
   `data/trace_sets/hca_compress_c128` and **nothing enforces it**. This should
   be an assertion in the orchestrator.
2. **Retries wrap the whole call, not the request.** See §3.2.

The loop itself is `KernelGenerator`: one `get_prompt`, then per round evaluate →
`get_optimization_prompt` → regenerate, 10 rounds by default. Prompts are
**stateless resends** — the full Definition plus current code plus the evaluation
log go back in every round — which is why a dropped reply costs a full round's
tokens.

### 3.4 Step 4: report

**This does not exist yet** — there is no report generator in `tools/`, and the
plan asks for one. It is the only entirely new component on this side.

The requirements from the plan, plus what the repository's own measurements say
the report must not do:

- **Rank on achieved bandwidth against measured peak, not `speedup_factor`.**
  This is the single most important design constraint on the report and it has
  its own memory note: `speedup_factor` is solution latency over a chain of eager
  PyTorch ops, which at small sizes measures op-dispatch overhead and at large
  sizes saturates, so "a kernel can be 30x off roofline and still report a
  respectable 1.2x". Peak must be **measured** (d2d copy, `roofline.py`
  `measure_peak_bandwidth`) — the delivery README is explicit that the 4800 GB/s
  spec figure "is paper — do not use it", and `HCA.md` was carrying that exact
  error until it was corrected. Measured peak is 4218 GB/s.
- **State which byte model produced the bandwidth**, because a task with a
  `bytes_model.py` and one without are not comparable, and the fp32-staging
  correction that `bytes_model.py` now applies to two of five solutions shifts
  the ranking by 1.6%.
- **A failure taxonomy, not a zero.** The plan's "如果有模型生成的kernel不可用
  导致没落盘等原因，追加说明" should be *structured*, because the distinctions
  matter: `SUCCESS`, `DROPPED` (gateway returned nothing, billed anyway),
  `UNUSABLE` (generated code does not extract or does not build), `INCORRECT`
  (built, wrong answers), `TIMEOUT`. An `INCORRECT` kernel that ran fast is not a
  zero-speedup kernel, and collapsing them loses the most interesting signal.
- **Exclude the small-workload regime from the ranking** and say why, per §2.2.
- **Report the measurement conditions** — device, SM count, measured peak, byte
  model identity — so the table is reproducible rather than a claim.

For the HCA task the report's expected shape is already knowable and makes a good
acceptance test for the generator: `claude_opus_5` ~600 GB/s (14.2% of peak),
`qwen3_8_max` ~1988 GB/s (47.1%), `gpt_6_astra` ~3294 GB/s, `torch_compile`
~1438 GB/s, `triton_h200` ~222 GB/s.

## 4. Concrete gaps between this design and the repository

Listed so they can be turned into work items rather than rediscovered.

| # | Gap | Where |
|---|---|---|
| 1 | No orchestrator; steps are chained by hand | new `tools/run_pipeline.py` |
| 2 | No `task.yaml`; paths and identities are module constants | `tools/*.py` (e.g. `TASK`, `DEF_NAME`, `ROOT` in `gen_solution_llm.py`) |
| 3 | No `models.yaml`; four env files × seven aliases, unread contract | new |
| 4 | No report generator | new |
| 5 | Anti-hack checks (C2) do not exist in any form | `tools/verify_task.py` |
| 6 | Input-realism is not a declared, checkable property | `tools/verify_task.py` + `task.yaml` |
| 7 | Generation root and measurement root can silently diverge | enforce in the orchestrator |
| 8 | Four schema constraints live in prose (`HCA.md` §2) | promote to `verify_task.py` checks |
| 9 | `validate.py`'s `benchmark` check carries a hardcoded `BenchmarkConfig` | reconcile, or document the divergence |
| 10 | No token accounting; `gen_solution_llm.py` extracts `usage` and drops it on the floor | optional, but it is why run cost is unanswerable |
| 11 | Reference/native reconciliation (C2, `HCA.md` §8 item 9) is open | gate item, currently prose |

## 5. Proposed layout

New components only; everything else is as it already stands.

```
kernel_pipeline/
├── docs/
│   ├── architecture.md            # this file
│   ├── contracts.md               # the contract catalogue (C1–C4 + verifiers)
│   └── op-types/                  # per-op notes (HCA.md §8 item 1 wants one)
├── tasks/<task>/
│   ├── task.yaml                  # C4: identity, target, roots, model provenance, split
│   ├── PROVENANCE.md              # C2/C3: exact-vs-synthesised, pinned revisions
│   └── ...                        # Definition, solutions/, workloads, eval_config.yaml
└── tools/
    ├── verify_task.py             # C1–C3 gate (extends validate_task.py)
    ├── run_pipeline.py            # stage 1 orchestration, 1.1→1.5
    ├── run_traces.py              # stage 2 orchestration, 2.1→2.4
    ├── models.yaml                # model configuration contract (§3.1)
    └── report_traces.py           # the markdown report (§3.4)
```

## 6. Summary of the improvements over the plan as stated

The plan's two modules and their step lists are, with one exception, the right
decomposition and this document keeps them. What changes:

1. **The 4 contracts get names and a verifier.** The plan asks for "四组contract";
   `contracts.md` says which four, what each asserts, and which checks implement
   it. Two of the four (anti-hack, input realism) do not exist in the codebase at
   all today — that is the largest real gap.
2. **"Ready for trace" must be a blocking gate, not a status.** Otherwise a
   defect surfaces at stage 2 and is attributed to the model instead of the task.
3. **The gate is split into mechanisable and human.** "符合实际工程语义" is not a
   check the pipeline can run; a gate that claims otherwise is worse than one
   that reports the open question.
4. **Search evidence carries a trust level.** The repo's own history is a list of
   plausible descriptions that were wrong; the pipeline must encode that leads
   become facts only by execution and pinning.
5. **Input realism is a criterion, not a resource question.** The plan's
   "如果有资源则直接生成，如果没有则拉取最小化可运行模型" is right, but the
   decision that matters is *whether the op's inputs are physically constrained*,
   and the repo already has the third path (probe exact tensors, synthesise the
   activation) that should be the default.
6. **The report ranks on bandwidth vs measured peak, and reports failures as a
   taxonomy.** Ranking on `speedup_factor` and collapsing failures to zero are
   both measured mistakes in this repository's history.
7. **The generation root and the measurement root are asserted equal.** One line,
   and it closes a silent-wrong-answer hole.
