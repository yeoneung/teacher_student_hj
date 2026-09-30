# Running the experiments

This directory contains the main and supplementary experiments for
*Information loss and minimal feedback in teacher-student policy learning:
a Hamilton-Jacobi analysis*, together with their numerical analyses and
computational dependencies.

## Stored results

From the repository root, run `python download_artifacts.py` and
`python verify_artifacts.py`. The latter verifies file integrity and links
all 1,306 reported policies to saved checkpoints and per-run evaluations.
Do not overwrite these archived observations with new training output.

Work in a separate writable reproduction copy for the analyses below. Run
commands from its `research/` directory in the recorded Python/CUDA environment.

```sh
python analyze_minimal_teaching.py
python analyze_neural_information.py
python analyze_direct_queries.py
python current_method_analysis.py
python analyze_improving_sets.py
python impact_reporting.py
python verify_evidence.py --stage-inputs
```

The first three analyses re-evaluate saved neural policies on CUDA. The next
three compute statistics from saved outcomes. `verify_evidence.py` stages the
two reference ZIPs from the repository root into `reference_inputs/` and checks
policy eligibility, numerical accounting, calibration and reference hashes.
The analysis scripts write JSON/CSV numerical reports.

## New main experiments

Use another copy with an empty `results/` directory. Run one timed GPU training
process at a time. Configurations, seeds and update budgets are in the scripts.

```sh
python verify_minimal_information.py
python verify_approximate_feedback.py
python minimal_teaching_study.py
python neural_information_transfer.py
python neural_information_confirmation.py
python direct_query_study.py
```

The confirmation requires the initial neural report. Direct-query comparisons
reuse designated minimal-feedback caches. Run the three corresponding analyses
after training. GPU and library versions can affect timing and numerical results.

## Supplementary studies

| Entry point | Role |
|---|---|
| `impact_core.py` | Exact teacher-continuation diagnostics and calibration |
| `impact_learning.py --stage mechanism` | Supporting teacher-quality/noise study |
| `impact_refinement.py --stage pilot` | Validation-based coefficient selection |
| `impact_replication.py --stage replication_v3` | 50 fitted policies for the continuation-audit comparison |
| `impact_noise_validation.py` | 4,800 local-action diagnostics |
| `impact_external_checks.py` | Additional reference checks retained by the evidence audit |
| `improving_sets_study.py` | 24 exploratory analytic-base fits |
| `improving_sets_warm.py` | 12 exploratory pretrained-student fits |
| `improving_sets_replication.py` | 20 independent fits using the validation-selected parameter |

Reference teachers and initial students come from the archived inputs. For new
supplementary runs, first stage the references in a reproduction copy, then set
`impact_core.PROJECT` to that copy's `research/reference_inputs` before importing
and running a study. The same setting is required before
`improving_sets_study.py`, `improving_sets_warm.py` or
`improving_sets_replication.py` loads a reference policy. For example, from
`research/` in the reproduction copy:

```python
from pathlib import Path
import runpy
import impact_core
impact_core.PROJECT = Path("reference_inputs").resolve()
runpy.run_path("improving_sets_study.py", run_name="__main__")
```

Preserve the supporting calibration and selection protocols when reproducing
later stages. Run the set pilot before the warm study and the warm study before
independent replication. These studies have different budgets and samples;
their results must not be pooled or attributed to a different training rule.

## Algebraic checks

`teaching_information_loss.py` checks the exact information-loss and shared-fit
constructions. `verify_minimal_information.py` checks 120 recovery and
indistinguishability cases. `verify_approximate_feedback.py` checks 2,304
alignment/noise cases. `verify_improving_sets.py` checks projection geometry,
gradients and the exact quadratic policy bound. The last script uses CUDA for
its quadratic-control check; the other three use the CPU.

The original sources and all outcomes are retained in release v1.0.0.
The current dependency record contains only computational modules retained here.
