# Teacher-student learning through Hamilton-Jacobi information

Experimental code and data for *Information loss and minimal feedback in
teacher-student policy learning: a Hamilton-Jacobi analysis*.

Hamilton-Jacobi policy evaluation identifies cost information lost when a
teacher action meets an input constraint. The experiments test which part of
that information a shared student needs and when restoring it improves learning.

## Experiments

| Paper component | Fitted policies | Experimental code | Numerical analysis |
|---|---:|---|---|
| Minimal-feedback grid | 480 | `research/minimal_teaching_study.py` | `analyze_minimal_teaching.py` |
| Direct-query and shared-actuation comparisons | 120 | `research/direct_query_study.py` | `analyze_direct_queries.py` |
| Neural information-restoration study | 240 initial + 360 confirmation | `research/neural_information_transfer.py`, `neural_information_confirmation.py` | `analyze_neural_information.py` |
| Nonlinear continuation-audit study in the supplement | 50 | `research/impact_replication.py` | `current_method_analysis.py`, `impact_reporting.py` |
| Improvement-set studies in the supplement | 24 + 12 exploratory, 20 independent | `research/improving_sets_study.py`, `improving_sets_warm.py`, `improving_sets_replication.py` | `analyze_improving_sets.py` |

These are **1,306 fitted policies**, with separate samples and statistics.
The release also preserves calibration, selection pilots, reference teachers,
diagnostic arrays and protocol records required to interpret these studies.
Exact constructions and algebraic checks are provided separately and are not
counted as neural training replications. See [execution instructions](research/README.md).

## Obtain and verify the data

The [v1.0.0 release](https://github.com/yeoneung/teacher_student_hj/releases/tag/v1.0.0)
contains the complete checkpoints, per-run evaluations, cached queries and
numerical reports. The three ZIP files are pinned by SHA-256 in
`artifact-manifest.json`.

```sh
python download_artifacts.py
python verify_artifacts.py
```

The downloader stages the numerical results under `research/results/` and
preserves the current code. To check every archived file, including the frozen
sources, run `python verify_artifacts.py --archive-dir .`.
No GitHub token is needed to download public data.

`verify_artifacts.py` checks the current code, all stored result files and the
correspondence between each of the 1,306 reported policies and its checkpoint
and evaluation bundle. It does not execute checkpoints or train policies.
The study-specific analyses perform the numerical re-evaluations.

## Environment and code

The recorded environment is Python 3.10.18, PyTorch 2.7.1+cu118, NumPy 2.2.5,
SciPy 1.15.3 and an NVIDIA RTX 4080 SUPER. Supplementary checks also use OSQP
and threadpoolctl. The full historical environment record is in
`research/environment_versions.json`.

Frozen training code and required computational modules retain their original
bytes. Current analysis scripts produce numerical reports; `code-manifest.json`
records their hashes and the corresponding source hashes in the release.
The cited v1.0.0 release remains unchanged as the fixed record of the experiments.
