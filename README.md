# Teacher-student learning through Hamilton-Jacobi information

Research materials for **Information loss and minimal feedback in teacher-student policy learning: a Hamilton-Jacobi analysis**.

The study asks which future-cost information a constrained teacher action can lose, and which part a fixed student output space needs. Under an explicit quadratic value-difference interface, the required exact scalar-query count is `rank(P.T @ Gamma)`. The rank proof uses classical linear recovery; the contribution is its teacher-constraint/student-representation interpretation and controlled evidence.

## Findings and limits

- Exact examples have identical improving targets but opposite cost effects after globally optimal shared-student fitting.
- Direct student-basis queries and a simple hybrid also recover the cost model exactly. All random-embedding caches have `r = min(k,m)`, so relevant queries have **no count advantage over hybrid** there.
- In a constructed shared-actuation geometry, `k=3, m=4, r=1`: relevant feedback needs one query, hybrid three, and the student basis four, with matching neural learning. This is a constructed case, not a naturally observed application benchmark.
- The 68.03% box-query saving against full-normal reconstruction is also achieved by hybrid. The cheap analytic oracle gives no runtime advantage for rank construction.
- A separate loss intervention confirms a conditional reduction in local Bellman regret despite higher action MSE. Effects are heterogeneous; adverse conditions and uncertain closed-loop mean effects are retained.
- A spectral upper bound and 2,304 algebraic checks address perturbed alignment and response error. No noisy-network or general control-superiority claim is made.

## Code and complete data

The repository contains browsable numerical sources, protocols and aggregate reports. The complete checkpoints, cache arrays and per-run outcomes are in [Reproducibility materials v1.0.0](https://github.com/yeoneung/teacher_student_hj/releases/tag/v1.0.0):

| Download | Contents |
|---|---|
| [research-artifacts-v1.zip](https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/research-artifacts-v1.zip) | Numerical code, simulation results, checkpoints and analysis reports under `research/` |
| [07_supporting_evidence.zip](https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/07_supporting_evidence.zip) | Reference numerical evidence for the nonlinear audit |
| [08_data_and_checkpoints.zip](https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/08_data_and_checkpoints.zip) | Reference policies and data required by the audit verification |
| [artifact-manifest.json](https://github.com/yeoneung/teacher_student_hj/releases/download/v1.0.0/artifact-manifest.json) | SHA-256 checksums for assets and all research files |

Extract `research-artifacts-v1.zip` at the repository root. Keep both reference ZIPs there. Check hashes against `artifact-manifest.json` before extracting. Internal editorial/design notes and submission packaging helpers are excluded from the public research bundle; frozen numerical sources and outcomes are preserved.

## Environment and verification

Recorded environment: Python 3.10.18, PyTorch 2.7.1+cu118, NumPy 2.2.5, SciPy 1.15.3, Matplotlib 3.10.9, OSQP 1.0.4, PyMuPDF 1.28.0, Pillow 11.0.0 and an NVIDIA RTX 4080 SUPER. Full versions are in `research/environment_versions.json`.

The CUDA analyses reevaluate stored policies without training. Run from `research/` after obtaining the complete artifacts:

```text
python analyze_direct_queries.py
python analyze_minimal_teaching.py
python analyze_neural_information.py
python analyze_improving_sets.py
python verify_evidence.py --stage-inputs
```

These analysis scripts regenerate manuscript figures and tables under `source/`. Use a separate reproduction copy to retain the supplied artifacts unchanged. Fresh training must use empty result directories in that copy: the drivers enforce frozen protocol/source hashes and retain every final checkpoint. Some studies depend on preceding reports, particularly the independent confirmation and set replication. See `research/README.md` for the order.

| Study | Fitted policies |
|---|---:|
| Nonlinear continuation audit | 50 |
| Improvement-set comparison | 56 |
| Neural loss intervention | 600 |
| Minimal-query grid | 480 |
| Direct-query controls and shared actuation | 120 |
| Total, without pooling statistics | 1,306 |

## Status

This repository provides research evidence and a manuscript under preparation. It does not assert journal acceptance. Authorship and submission declarations are maintained with the manuscript; public access does not itself certify those declarations.
