# Research code and reproducibility

Materials for **Information loss and minimal feedback in teacher-student policy learning: a Hamilton-Jacobi analysis**. The core question is which cost information constrained teacher actions discard and which part their student needs. The exact constructions in `source/current/information_loss.tex` and `joint_realizability.tex` isolate loss of normal-cone cost information and incompatibility of statewise improvement sets within a shared student class. They do not identify these causes in the nonlinear experiments. `teaching_information_loss.py` verifies closed-form witnesses, exact Bellman realization, scalar global fits and regularized examples, and renders the analytical figure. It performs no neural training. Its report is `results/teaching_information_loss.json`.

The active paper is `source/manuscript.tex` and `source/current/`. It reports a 480-fit minimal-query grid, 120 direct-query checks, a separate 600-fit loss intervention, continuation-audited teaching (50 fits), and improvement-set supervision (36 exploratory plus 20 independent fits). Results from these different procedures must not be pooled or attributed to the other algorithm.

The set teacher supplies an advantage upper model and a shared realized feasible anchor. The student learns squared distance to its improvement sublevel set, recomputing the current action projection at every update. The implementation uses epsilon=0, scalar quadratic curvature and a ball input constraint. Nonlinear curvature estimates are empirical, not certified.

| Entry point | Purpose |
|---|---|
| `improving_sets.py` | Two-ball projection and captured current-action set loss |
| `verify_improving_sets.py` | Independent SLSQP/gradient checks and exact LQ policy bound |
| `improving_sets_study.py` | 24 analytic-base exploratory runs, widths16/64 |
| `improving_sets_warm.py` | 12 pretrained-student exploratory runs |
| `improving_sets_replication.py` | 20 independent reaction runs at validation-selected alpha=.25 |
| `analyze_improving_sets.py` | Frozen hash/cost verification, paired statistics, diagnostics, figures |
| `current_method_analysis.py` | Separate 50-run continuation-audit analysis |

Run numerical entry points only in a separate reproduction copy to preserve supplied raw evidence and source locks. Run `analyze_improving_sets.py` to verify/reanalyze without training. The replication source checks the warm pilot's validation-only decision. Its five teacher/data seeds are disjoint from that pilot; all pilot outcomes remain available. Set supervision does not establish a cost advantage: mean cost .902697 versus .907593 point/adaptive and .891263 upper model, with paired intervals crossing zero.

All studies use PyTorch CUDA on an RTX4080SUPER, one timed training process at a time. Set comparisons match 1024 updates and report charged computation; they are not equal-wall-budget comparisons. Exact tests use CUDA for LQ and CPU SciPy as an independent projection reference.

## Controlled neural information transfer

`neural_information_transfer.py` contains the frozen initial 240-run protocol
(5 seeds, radii 0.5/2/8/32, widths 2/4/16, four losses).
`neural_information_confirmation.py` imports that unchanged implementation
and freezes 360 additional fits on ten disjoint seeds, radii 0.5/2/32,
the same widths, losses, data law and 4096-update budget. The confirmation
primary is radius0.5/width2, selected from the initial secondary response;
the initial radius2/width2 primary remains inconclusive. These are separate
samples, not pooled evidence. Both studies are complete.

The plant is linear with an exact zero-teacher value; the student is an
actually trained nonlinear SiLU network. The normal-restored loss equals
exact quadratic Bellman regret. Width16 has a known exact representation;
width2/4 constrain the output span. The experiment tests a controlled
information-loss intervention, not general control-algorithm superiority.

From this directory, run `analyze_neural_information.py` in the recorded
CUDA environment to verify all source/checkpoint hashes, re-evaluate all
600 policies, check bitwise parameter equality at inactive constraints,
and regenerate paired statistics, exact radial decomposition, CSV, table
and figure. No training occurs in that analysis. Raw policies and per-state
evaluations are in `results/neural_information_transfer/` and
`results/neural_information_confirmation/`. The confirmation plan records
both numerical source hashes and the complete parent report hash before
training. `results/neural_information_analysis.json` and
`results/neural_information_all_runs.csv` retain all conditions and losses.

Preserve these directories and frozen numerical sources. For a fresh
reproduction, copy the submission tree beneath `submission/reproductions/`
and use empty result directories there; run initial training before the
confirmation driver so its parent hash is well defined. Do not run multiple
timed CUDA training jobs together. These protocols match updates, and do
not support an equal-wall-time speedup claim.

# Research workspace

## Student-relevant minimal scalar teaching information

`minimal_teaching.py` defines the restricted value-difference oracle and
the relevant/gain/random/full-face/full-vector interfaces.
`minimal_teaching_study.py` is frozen by
`results/minimal_teaching_study/protocol_lock.json` before 480 fits:
5 seeds × 2 action dimensions × 2 fixed student dimensions × 2 constraints
× 2 radii × 6 interfaces. It uses one sequential CUDA process, 4,096
updates per fit, paired initialization/data/minibatches and final checkpoints
without validation selection. Full vectors use a privileged interface and
are not equated to a scalar-query count. Analytic plant access is restricted
at the receiver interface; the theorem is not computational hardness of
the disclosed model.

Run `analyze_minimal_teaching.py` in the recorded CUDA environment to
verify source/cache/checkpoint hashes, reevaluate every policy and export
complete paired statistics and manuscript tables/figures without training.
`results/minimal_teaching_analysis.json` retains all 16 conditions and
six methods, including secondary intervals. Primary coefficient recovery
and exact query counts pass throughout; all 80 relevant/full pairs meet
the prespecified regret fidelity tolerance. The box query reduction is
68.03% relative to complete normal-space reconstruction, not to analytic
gradient access or wall time. No additional training remains active.

`verify_minimal_information.py` separately generates 120 algebraic
indistinguishability and response-noise checks, with raw arrays under
`results/minimal_information_exact/`. These are not learned policies.
The interface also checks 576 recovery cases through dimension128.
Preserve the frozen sources and result directories. For a fresh training
reproduction, use an empty result directory in a separate copy below
`submission/reproductions/`; never overwrite the supplied evidence.

All revised manuscript files, new code, results and packaging work are inside `submission/`. The numerical sources and their protocol hashes are frozen after their stated experiments. Later edits to analysis or packaging do not change those numerical sources.

The completed studies are:

| Entry point | Evidence | Role |
|---|---|---|
| `impact_core.py` | `results/exact_diagnostic/` | Complete student-trajectory teacher advantages and error accounting |
| `impact_learning.py --stage mechanism` | `results/mechanism/` | One-seed developmental teacher-quality/noise grid, 120 phases |
| `impact_refinement.py --stage pilot` | `results/pilot/` | 14-phase pilot and validation-based fixed-coefficient selection |
| `impact_replication.py --stage replication_v3` | `results/replication_v3/` | Selected audit rule, five independent student/data fits per task/method, 50 phases |
| `impact_noise_validation.py` | `results/noise_validation/` | 4,800 local actions; no student retraining |
| `impact_external_checks.py` | `results/factorial/`, `results/building/` | 114 factorial rows and 64 QP states with separate latency measurements |

`environment_versions.json` records the actual package versions. Experiments used Python 3.10.18, PyTorch 2.7.1+cu118, CUDA and an RTX 4080 SUPER. NumPy, SciPy, OSQP, threadpoolctl and Matplotlib are required; PyMuPDF and Pillow support manuscript checks. Training uses wall budgets, so fresh timing-based reruns need not produce bitwise identical selected policies.

## Verify the supplied results without training

Extract `09_impact_research.zip` into the submission directory if this live `research/` folder is absent. Keep archives `07` and `08` beside it. From `submission/research`, use the configured environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '..\..\.venv-hj-cuda\Scripts\python.exe' verify_evidence.py --stage-inputs
```

On another installation replace the executable path with an environment containing the recorded dependencies. The helper stages the original reference files under `research/reference_inputs/`, validates archive paths, and reads those files instead of the original project. It verifies frozen source hashes, selected policy hashes and eligibility, finite values, per-state cost means, local accounting, noise counts, QP solves and original evidence hashes. It does not rerun GPU learning or establish a population guarantee.

`impact_reporting.py` regenerates the summary and manuscript tables/figures in the current project layout. For staged reference inputs, set `impact_core.PROJECT` to the absolute `reference_inputs` path in the calling Python process before importing `impact_reporting`; no frozen source edit is needed.

## Fresh experimental reproduction

Preserve the supplied `results/` and protocol locks. Work in a separate copy of the submission directory under `submission/reproductions/<run>/`, retaining the `source/` and `research/` layout. Stage the reference archives there, then use `runpy.run_path` on a numerical entry point after assigning `impact_core.PROJECT` to that copy's `research/reference_inputs/`. Start with an empty result directory in the copy. Run the stages in the table's order, using only one timed GPU training process at a time. The pilot and exact diagnostic reports are dependencies of later stages; do not skip them or substitute test-based selection.

The completed initial mechanism and pilot failures are part of the evidence. Their scripts are not interchangeable with the final replication. The final controlled teacher is fixed: refresh replaces a 64-entry window while retaining the original other 4,032 entries. The adaptive reference changes the coefficient within the same fixed-teacher loop. It is not a reproduction of the earlier complete evolving-teacher algorithm.

The complete new archive includes 184 saved policies and raw evaluation bundles. `vendor_provenance.json` maps the 116 unchanged numerical dependencies to their original hashes. `reference_manifest_v10.json` enables verification of original evidence shipped in archives `07` and `08`. No outside service or data repository is required for those artifact checks.


## Direct-query controls and approximate alignment

`direct_query_study.py` freezes and runs 120 additional GPU fits: 80 direct-interface checks reusing four original conditions, and 40 fits on a constructed shared-actuation geometry. It audits query counts and inclusive query-construction/response/decoding timings on 80 original caches plus five structural caches. `direct_query_controls.py` implements student basis, orthogonal student basis, hybrid and zero-direction-pruned hybrid without accessing hidden coefficients.

`verify_approximate_feedback.py` verifies 2,304 constructed spectral-truncation and response-noise cases, with no neural training. An aborted OpenMP precheck lock is retained for provenance; it produced no outcomes. The completed protocol uses only NumPy for these checks.

`analyze_direct_queries.py` reevaluates all 120 checkpoints, compares exact-recovery methods against paired full-information references, validates source/data hashes and writes `results/direct_query_analysis.json`. It also generates the direct-query, paired-effect, corrected witness and approximate-feedback figures. Original frozen numerical sources, outcomes and figures are preserved. Additional fits reuse the CUDA environment and training schedule specified above. No aggregate statistics pool this follow-up with independent loss-intervention samples.


Public repository: https://github.com/yeoneung/teacher_student_hj
Complete numerical assets: https://github.com/yeoneung/teacher_student_hj/releases/tag/v1.0.0
The public archive excludes internal editorial/design notes and packaging helpers. Numerical sources, protocols and outcomes retain their original hashes.
