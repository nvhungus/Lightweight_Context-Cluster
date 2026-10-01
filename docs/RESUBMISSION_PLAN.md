# HBCC resubmission plan

Working plan for revising *"HBCC: Hybrid Block Context Cluster for Lightweight Image
Classification"* (rejected at FAIR 2026) and resubmitting by **15 November 2026**.

Everything here is organised around the two reviews. Reviewer 1 gave Weak Accept (3) with
"low novelty"; Reviewer 2 gave Weak Reject (2) and is the reviewer whose objections decided the
outcome. Every objection R2 raised turned out to be correct when checked against the code.

- Code and runs: branch `resubmission`, github.com/nvhungus/Lightweight_Context-Cluster
- Experimental protocol (the source for Table II): [`docs/protocol.md`](protocol.md)
- Every run by ID: [`configs/run_plan.yaml`](../configs/run_plan.yaml); launched with
  `tools/run_queue.py`, aggregated with `tools/aggregate_seeds.py`
- Run IDs below (A1, B3, …) are the IDs in that file. The A/B letters are labels only; any run
  can go on either Kaggle account, except that a KD run must run on the account that trained its
  teacher.

**Status key:** ✅ done · ⏳ running or queued · ⬜ pending · ⚠️ decision needed

---

## 1. Where things stand

### Code (all merged, no training-code changes since commit `72070fe`)

| Change | What it does |
|---|---|
| Seeding | `train.seed` defaults to 42; seeded DataLoader generator and worker init; optional `train.deterministic` |
| Provenance | Every run writes `run_info.json`: git commit and dirty flag, full command, seeds, torch/CUDA/cuDNN versions, GPU and CPU model |
| Paired tests | Every run saves `test_predictions.npy` / `test_targets.npy`, so McNemar tests between runs are possible |
| Cost profiling | Peak memory measured **per batch size**; CUDA-Graph latency with an automatic equality check against eager; clustering-op MACs that fvcore does not count; median of repeated timings |
| New ablation operator | `conv3x3` local branch (learned full 3×3), for the LBPConv comparison R2 asked for |
| Ablation configs | 9 one-change variants of HBCC-Medium (see §2, R1.3) |
| Food-101 code | Vendored at `food101/` from `banlanhat69/Lightweight-Context-Cluster@bedafe4` — the exact code that produced the Food-101 numbers. See `food101/VENDORED.md` |
| Run tooling | `tools/run_queue.py` (run by ID, same-seed KD teachers, session deadline guard), `tools/aggregate_seeds.py` (mean ± std, paired t-test, exact McNemar) |

### Runs completed

**Smoke test** (Kaggle T4): all 22 configurations train for one epoch on the real datasets, write
every artifact, and find their teachers. Unit tests pass on Kaggle.

**Tiny-ImageNet, seed 42, 100 epochs** (sessions A_s1 and B_s1, 8/8 runs clean, all on commit
`72070fe`, identical test set and order):

| Model | Params | Best val top-1 | Test top-1 |
|---|---|---|---|
| ResNet-18 | 11.27M | 58.70 | 58.81 |
| **HBCC-Medium** | 1.80M | **56.43** | 56.18 |
| ShuffleNetV2 | 1.46M | 55.98 | 55.85 |
| **All-cluster Medium** (ρ=0) | 1.79M | **55.84** | 55.57 |
| HBCC-Small | 1.32M | 55.27 | 55.32 |
| HBCC-Medium, stem stride 2 | 1.80M | 50.68 | 50.80 |
| CoC-baseline | 2.46M | 49.45 | 49.56 |
| MobileNetV2 | 2.48M | 49.25 | 49.49 |

Three findings that shape the revision:

1. **Stem stride is by far the largest effect measured so far**: stride 1 → stride 2 costs
   **−5.75 pp**. This is the strongest support yet for the token-degeneracy / resolution rule.
   (Caveat: stride 2 also cuts activations ~4×, so resolution and cluster occupancy *r* are
   confounded; the *r* sweep B8/B9 separates them at fixed resolution.)
2. **The hybrid block's advantage over pure clustering is small**: Δ = +0.59 pp on validation at
   seed 42 (McNemar on test p = 0.11, not significant). This is the **gate** — see §4.
3. **HBCC-Small CE is behind ShuffleNetV2** (55.27 vs 55.98) at a similar parameter count. The
   first submission claimed both HBCC CE models beat ShuffleNetV2; that no longer holds.

Comparing the six re-run configurations against the first submission (same protocol, only the
seeding code changed) gives a first estimate of run-to-run noise: roughly **0.3–0.5 pp**.

### Cost measurements on T4 (partly done)

- **Confirmed:** the old paper's "Mem" column was the **batch-128** peak printed next to batch-1
  latency. At batch 1, HBCC uses 16.8 / 19.5 MB, comparable to MobileNetV2 (17.7) and
  ShuffleNetV2 (14.0) and ~3× below ResNet-18 (57.4).
- **Confirmed:** eager batch-1 latency is ~85 % CPU kernel-launch overhead. The torch profiler
  reports ~1.0 ms of actual GPU work for HBCC versus 1.7 ms for ResNet-18.
- **Confirmed and unchanged:** MACs match the first submission exactly; the clustering operations
  omitted from the old count add < 1 %.
- **Still a real disadvantage:** at batch 128 HBCC is slower and heavier than every baseline
  including ResNet-18 (39.3 vs 27.5 ms; ~480 vs ~326 MB), because of the Stage-1 channel
  expansions at full 32×32 resolution.
- **To redo:** the CUDA-Graph latencies and batch-128 memory from the first measurement were
  affected by two profiler bugs, both fixed. One ~30-minute session re-measures them (M1).

### Writing tasks identified

| ID | Task |
|---|---|
| W1 | Text corrections: depthwise saving is 1/C_out + 1/k² (not 1/k²); cite MetaFormer; add a venue to the Hinton reference; PointReducer is 3×3 stride-2 (not 1×1); coordinates are in [−0.5, 0.5] (not [0, 1]); Fig. 1 caption colours are swapped |
| W2 | Rewrite the LBPConv description (§II-D, §III-F) to match the code |
| W3 | Stage 4 is similarity-gated global aggregation, not clustering (§III-G) |
| W4 | Food-101 variant details (§III-I): layer-scale init is 1e-3; define the cluster-balance loss; dropout placement; cite the vendored code |
| W5 | Moderate the claims: limit RQ1 and "lightweight" to parameter and batch-1-memory efficiency; fix the Fig. 4 caption; delete "fine-grained", "scales with class-space complexity", the Food-101 "upper bound" argument, and the untested "clustering exploits soft distributions" explanation; state that the Tiny-ImageNet test split is the official validation split; draft Limitations |

---

## 2. Work organised by reviewer point

### R1.1 — Multiple runs and statistical analysis

> "conduct multiple independent runs and report appropriate statistical analyses", especially
> "where the performance differences … are small"

**Plan.** Three seeds (42, 43, 44) for the Tiny-ImageNet main table: ResNet-18, ShuffleNetV2,
HBCC-Small, HBCC-Medium (CE and KD), all-cluster, ShuffleNetV2+KD. Report mean ± std, a paired
t-test across seeds and an exact McNemar test per seed on the shared test images. MobileNetV2 and
CoC-baseline stay at one seed (6–9 pp behind). CIFAR-10 and Food-101 stay at one seed, stated as a
limitation.

**Covered by:** A2/A3/A12/B1/B2/A1 ✅ (seed 42) · A10–A14 ⏳ · A18–A22 ⏳ · `tools/aggregate_seeds.py` ✅

### R1.2 — Clarify the "lightweight" claim and deployment capability

> "although HBCC uses fewer parameters and MACs, it still exhibits higher latency and memory
> consumption than some lightweight CNNs"

**Plan.** Report cost per batch size, both eager and CUDA-Graph latency, with GPU and CPU model
stated. Claim parameter efficiency and batch-1 memory efficiency only; concede MACs and
large-batch cost. Say "single-image GPU inference", not "edge device" — nothing was measured on
edge hardware.

**Covered by:** M1 re-measure ⬜ · W5 ⬜ · partial data already in hand ✅

### R1.3 — More comprehensive ablations

> "quantify the individual contributions of key components, including the local branch, the
> clustering mechanism, fold partitioning, stem stride, and knowledge distillation"

| Component | Ablation | ID | Status |
|---|---|---|---|
| Local branch | ρ=0 at HBCC's stem and depth | A1, A14, A22 | ✅ seed 42 · ⏳ seeds 43/44 |
| **Clustering mechanism** | all-local (clustering removed) | **B22** | ⬜ — see §3(a) |
| Fold partitioning | no fold (global clustering everywhere) | B6 | ⬜ |
| Stem stride | stride 2 at equal depth | B10 | ✅ (−5.75 pp) |
| Knowledge distillation | CE vs KD, same architecture | A6/A7 vs B1/B2 | ⏳ |
| (extra) cluster occupancy *r* | r = 4 and r = 1 at fixed resolution | B8, B9 | ⬜ |
| (extra) Stage 4 | 2×2 proposals instead of 1 centre | B7 | ⬜ |

### R1.4 — Fairness and scope of the baseline comparison, especially Food-101

> "the number of baselines is limited and the training recipes are not fully consistent across
> models"

**Plan.** Two options, decision deferred until the gate is settled:

- (a) Run the shared-recipe control that already exists in the vendored code
  (`food101/configs/food101/base_fair_150.yaml`, both models, 150 epochs) — ~34 GPU-hours.
  Optionally add MobileNetV2 and ShuffleNetV2 at 224 px — ~30 more.
- (b) Present Food-101 as a secondary "scales to 224 px" result and state the recipe confound and
  the single baseline as limitations.

**Covered by:** T9 ⚠️ decision needed · W4/W5 ⬜

### R1.5 — Moderate the claims

> "avoiding overly broad generalizations based on a limited number of benchmarks and relatively
> small performance margins"

**Covered by:** W5 ⬜

---

### R2.1 — Knowledge distillation is applied only to HBCC

> "On CIFAR-10 the baselines are CE-only while HBCC is reported only with KD; HBCC CE results on
> CIFAR-10 are missing. Please report HBCC CE on all datasets, and ideally the baselines with the
> same KD."

**Plan.** Report HBCC CE on every dataset, and train ShuffleNetV2 and MobileNetV2 with the same
teacher, temperature and α. Each KD student uses the ResNet-18 teacher of the **same seed**.

**Covered by:** B15, B16 (CIFAR HBCC CE) ⬜ · A8, A9, A17, A25, B20, B21 (baseline KD) ⬜

### R2.2 — The CoC baseline is not a clean ablation

> "It uses stem stride 2 … which by the paper's own analysis gives one point per cluster at
> Stage 4. The 7–10 pp gap therefore mixes the local branch with stem stride, depth and fold
> design. A proper ablation would keep HBCC's stem, folds and depth and set rho=0, and would also
> compare LBPConv against a learned 3×3 and remove channel shuffle."

| Sub-point | Ablation | ID | Status |
|---|---|---|---|
| (a) clean ρ=0 | all-cluster at HBCC's stem/folds/depth (1.79M vs 1.80M params) | A1, A14, A22 | ✅ seed 42 · ⏳ |
| (b) LBPConv vs learned | learned depthwise; learned full 3×3 | B3, B4 | ⏳ |
| (c) channel shuffle | shuffle removed | B5 | ⬜ |

The existing CoC-baseline is **reframed** in the text as a CoC-style reference configuration, not
as a controlled ablation.

### R2.3 — Efficiency claims do not match the numbers

> "HBCC has 3–5× the MACs of MobileNetV2 and ShuffleNetV2, is slower … and uses 3–5× the peak
> memory. … The Fig. 4 caption ('Pareto-dominant on both the parameter and latency axes') is
> incorrect for latency."

**Plan.** The memory part of this objection rests on the mislabelled column and is answerable with
data already in hand. The MACs and large-batch parts are conceded. Fig. 4 is split into a
parameter axis and a latency axis with the latency mode and batch size stated.

**Covered by:** M1 ⬜ · W5 ⬜ · existing data ✅

### R2.4 — "Student beats teacher" is within seed variance

> "A 0.32 pp top-1 gain from a single run is within normal seed variance, and students matching or
> exceeding teachers is well documented in the KD literature. The explanation that clustering
> 'exploits soft distributions more effectively' than convolution is not tested."

**Plan.** Demote this from the headline regardless of outcome. Report it only if it survives the
mean over three seeds, with a paired test. Test the explanation directly by comparing the KD gain
for HBCC against the KD gain for ShuffleNetV2 and MobileNetV2 under the identical teacher and
recipe; drop the claim if the gains are comparable. Cite the KD literature R2 refers to — see
§3(b).

**Covered by:** A6/A7, A15/A16, A23/A24 (HBCC KD, 3 seeds) ⬜ · A8/A9, A17, A25 (baseline KD) ⬜

### R2.5 — Food-101

> "The two models use different optimizers, augmentation and regularisation, and the HBCC variant
> also changes normalisation, adds a Stage-4 DWConv branch and a cluster balance loss that is never
> defined. The claim that 0.38 pp is an upper bound … does not follow. Most importantly, MACs,
> latency and memory at 224×224 are not reported."

| Sub-point | Response | Status |
|---|---|---|
| (a) no costs at 224 px | MACs measured: **548M (HBCC-2.5M) vs 1.82G (ResNet-18)**, 3.4× fewer. T4 latency/memory via M2, < 1 h | ✅ MACs · ⬜ M2 |
| (b) recipes differ; GN, Stage-4 DWConv, balance loss undefined | Define the balance loss and justify the architectural differences in W4 (see §3(b)); shared-recipe control is T9 | ⬜ / ⚠️ |
| (c) "upper bound" claim | Delete it. The logs show HBCC had converged (best epoch 199/200, validation flat at 77.5–77.7 for >10 epochs), so the argument does not hold | ⬜ (W5) |

Also to report in the text: on Food-101 the validation gap to ResNet-18 is **−1.67 pp** while the
test gap is −0.38 pp. Both should be given, not just the favourable one.

### R2.6 — Method details

> "LBP thresholding is non-differentiable. How are gradients passed through the local branch?"
> "The fixed LBP filters are counted as parameters … In that case they do cost MACs, contradicting
> 'near-zero MAC cost'." "The reported HBCC MACs exclude the assignment and aggregation
> operations." "At Stage 4 a single center … reduces to global pooling rather than clustering."

R2 was reasoning from the paper and was right on all four counts. The **code is sound; the
description is wrong.** The fix is to the paper, not the code — changing the code would invalidate
every HBCC result.

| Sub-point | Reality in the code | Fix |
|---|---|---|
| (a) gradients | There is **no threshold**: 8 fixed difference filters (gₚ − g꜀), then BN, GELU and a *learned* 1×1 conv (8C → C). Fully differentiable | W2 ⬜ |
| (b) MACs and parameters | LBPConv costs **6.5M / 10.8M MACs (7–8 % of the total)**; fixed filters are 1,728 / 2,304 non-trainable parameters; the 1×1 projection is learned | W2 ⬜ |
| (c) clustering MACs excluded | Measured: **+0.4M, < 1 %**. The profiler now reports `macs_incl_cluster` | ✅ code · ⬜ text |
| (d) Stage 4 is pooling | Correct. Reword as similarity-gated global aggregation; test the alternative (2×2 proposals) | W3 ⬜ · B7 ⬜ |

Two further claims in §III-F and §III-C that the code contradicts and that must go: "near-zero MAC
cost", and the "soft histogram of texture patterns" argument for AvgPool (the activations are not
binary, and in a hybrid block the cluster centres never see the local branch's output — the two
branches meet only at the fuse convolution).

### R2.7 — Datasets and claims

> "Tiny-ImageNet is resized from 64×64 to 32×32, which discards information… The reported test set
> appears to be the official validation set; please state this. Tiny-ImageNet and Food-101 are not
> fine-grained benchmarks… the conclusion that HBCC's advantage 'scales with class-space
> complexity' is not supported."

**Plan.** State the resolution choice and the test-split identity explicitly (already written into
`docs/protocol.md`); delete the "fine-grained" framing and the class-complexity conclusion; keep
32 px as a stated limitation (a 64 px run was considered and cut for cost).

**Covered by:** W5 ⬜ · `docs/protocol.md` ✅

### Minor points

Fig. 1 caption colours swapped · depthwise saving is 1/C_out + 1/k² · MetaFormer cited · Hinton
reference needs a venue · tone down "most remarkable" and "decisive".

**Covered by:** W1, W5 ⬜

---

## 3. Open gaps

### (a) All-local ablation — B22

R1.3 asks for the contribution of the **clustering mechanism**, and nothing in the original plan
removed it. A1 removes the *local* branch; without its counterpart a reviewer can argue the
clustering contributes nothing and the result comes from the depthwise convolutions. This matters
more now that the hybrid gain at seed 42 is only +0.59 pp.

Config added: `configs/ablations/hbcc_medium_alllocal.yaml` — HBCC-Medium with every block's token
mixer set to the local operator on all channels (LBPConv in Stage 1, DWConv in Stages 2–4). Same
widths, depths, stem, MLPs and recipe. 1.84M parameters / 157.7M MACs, against HBCC-Medium's 1.80M
/ 139.3M and all-cluster's 1.79M / 125.6M.

**Status:** ⬜ queued as B22 (seed 42); conditional seeds C17/C18 if it lands within 1.0 pp.

### (b) Writing gaps with no experiment attached

1. **KD literature.** R2.4 notes that students matching or beating teachers is well documented.
   Cite it and position any such result as consistent with that literature rather than unique to
   clustering. Suggested: Furlanello et al., *Born-Again Neural Networks* (ICML 2018); Stanton et
   al., *Does Knowledge Distillation Really Work?* (NeurIPS 2021).
2. **Food-101 architecture justification.** R2.5 flags GroupNorm and the Stage-4 DWConv branch as
   unexplained. Ablating them would cost ~22 GPU-hours per run, which is out of budget, so write a
   short design justification (GroupNorm for the small batch of 32 at 224 px; the Stage-4 local
   branch to recover spatial detail at 7×7) and list the architectural difference between the
   32 px and 224 px models as a confound in Limitations.

**Status:** ⬜ both, writing only.

---

## 4. The gate: which framing the paper takes

A single pre-registered comparison decides how the paper is framed. Δ = best validation top-1 of
**HBCC-Medium CE** minus **all-cluster Medium CE**, on Tiny-ImageNet, per seed. The two models are
budget-matched (1.80M vs 1.79M parameters; HBCC has ~10 % more MACs).

| Seed | Δ (validation) |
|---|---|
| 42 | **+0.59** (test +0.61; McNemar p = 0.11) |
| 43 | pending (A13 vs A14) |
| 44 | pending (A21 vs A22) |

**Rule:** keep the hybrid-block framing only if **mean Δ ≥ 0.5 pp and Δ > 0 in all three seeds**;
otherwise pivot to the design-analysis framing.

- **Hybrid-block framing** — the paper keeps HBCC as the contribution, now with a budget-matched
  ablation behind it.
- **Design-analysis framing** — the contribution becomes the cluster-occupancy design rule and the
  stage-wise allocation study (the −5.75 pp stem-stride result is the lead), with the hybrid block
  as one component among several. Working title: *"Token Degeneracy and Stage-wise Allocation in
  Context-Cluster Classifiers"*.

The design-analysis framing is valid under either outcome, so drafting should start there. No
scheduled run depends on the gate — nothing is wasted either way.

---

## 5. Remaining sessions

Each Kaggle session is capped at 12 hours; the queue tool refuses to start a run that cannot
finish within 11. Measured Tiny-ImageNet run time is ~1.8–2.3 h per run regardless of model, since
training is CPU-bound (JPEG decode + RandAugment), so **4–5 runs per session**. Running two jobs on
the two T4s at once does *not* help — they compete for the same 4 CPU cores.

| Session | Runs | What it settles | Needs attached | ~Hours |
|---|---|---|---|---|
| **A_s2** | A14, A13, A10, A11, A12 | Gate seed 43; seed-43 main table; seed-43 KD teacher | — | 10.2 |
| **B_s2** | A22, A21, B3, B4 | Gate seed 44; LBPConv vs depthwise vs learned 3×3 | — | 8.9 |
| **A_s3** | A7, A6, A8, A9 | KD seed 42, HBCC **and** baselines (R2.1, R2.4) | A_s1 output (teacher A2) | 8.8 |
| **B_s3** | B22, B5, B6, B7 | Clustering mechanism; shuffle; fold; Stage 4 | — | 9.2 |
| **A_s4** | A15, A16, A17, A18 | KD seed 43; seed-44 KD teacher | A_s2 output (teacher A10) | 9.0 |
| **B_s4** | B8, B9, A19, A20 | *r* sweep (r = 4, r = 1); seed-44 baselines | — | 9.0 |
| **A_s5** | A23, A24, A25 | KD seed 44 | A_s4 output (teacher A18) | 6.7 |
| **CIFAR-10** | B11–B21, in 3–4 sessions | Replaces the whole CIFAR-10 table, which currently has no logs | the session that ran B11, for the KD runs | ~35 |
| **C-runs** | subset of C1–C18 | Extra seeds for ablations within 1.0 pp of HBCC-Medium | — | ≤ 13 |
| **M1 + M2** | — | Re-measure T4 cost at 32 px and 224 px | — | ~1 |
| **T9** ⚠️ | Food-101 shared recipe | R1.4 / R2.5b — optional, first to cut | — | ~34 |

CIFAR-10 run times are **not measured yet**, so its sessions are deliberately left unpacked; the
per-run time gets fixed after the first CIFAR session and the remaining sessions are packed then.
The current placeholder estimates (2.5–4.0 h) are conservative; if CIFAR behaves like
Tiny-ImageNet and is CPU-bound, every run will be ~2.5 h and the table fits in three sessions of
four runs. Order B11 (the teacher) before the KD runs B18–B21.

Totals excluding T9: **~19 GPU-hours** for A_s2 + B_s2, then **~77** for everything after them,
plus up to 13 conditional. At ~30 h per account per week, two accounts clear this in roughly two
weeks; one account alone takes about three and a half.

**Priority if quota is short:** the gate sessions (A_s2, B_s2) first, then the KD sessions (R2's
main objection), then the ablations, then CIFAR-10, then the conditional seeds. T9 is cut first.

---

## 6. Timeline to 15 November

| By | Milestone |
|---|---|
| Early Oct | Gate decision from three seeds; framing fixed; W1–W5 drafted |
| Mid Oct | KD sessions and the full ablation set finished |
| Late Oct | CIFAR-10 table; conditional extra seeds; M1/M2 cost re-measurement |
| **1 Nov** | **Experiment freeze.** No new runs after this except re-running a crashed job |
| 1–10 Nov | Aggregate all results; build tables and figures; write Results, Ablations, Discussion, Limitations |
| 10–14 Nov | Final checks (below); proofread; cover letter answering each reviewer point |
| **15 Nov** | **Submit** |

### Final checks before submission

- Every number in every table traces to a committed run directory with `metrics.jsonl`,
  `test_metrics.json`, `config.yaml`, `run_info.json` and `test_predictions.npy`.
- No number from the first submission survives unless it was re-run under `docs/protocol.md`.
  (The CIFAR-10 headline numbers of the first submission have **no training log** anywhere in the
  repository and are being replaced outright.)
- Table II matches `docs/protocol.md` exactly.
- Every accuracy cell is mean ± std over three seeds, or explicitly marked single-seed.
- Every KD run used the teacher of its own seed.
- Parameters include the fixed LBP filters; MACs include the clustering operations.
- Latency states eager or CUDA-Graph **and** the batch size; memory states the batch size.
- Every "beats" or "matches" claim carries a paired test, or is worded as "within noise".
- Figure captions match what the figures show (Fig. 1 colours; Fig. 4 axes).
- The released code reproduces one config per dataset from a clean clone.
