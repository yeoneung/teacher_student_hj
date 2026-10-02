"""Structural rank deficiency when several actuators share one student channel.

CPU-only geometry check. No training, no checkpoint use, no policy evaluation.

The teaching model, the feasible target and the rank of the projected normal
space are computed by the frozen interface in minimal_teaching.py, so the
definitions of the active face and of rank(P^T Gamma) are the ones used by the
query study. This file only supplies student output bases and tallies counts.

Student output bases, all orthonormal:
  ganged  - d/g channels, each driving a contiguous group of g actuators.
            g = 1 gives one channel per actuator and is the control.
  mirror  - even Fourier modes about node 0, so actuators i and d-i receive the
            same row of P under a reflection-symmetric policy class.
  random  - the embedding family used by the published query grid.

Counts per cached state, in the units of the query study:
  rank    = rank(P^T Gamma)        (the minimum of Theorem 2)
  hybrid  = min(k, m)
  student = m when any coordinate is active
  face    = k
"""
import hashlib
import json
from pathlib import Path

import torch

import minimal_teaching as mt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/shared_channels'

CFG = dict(
    shape='box',
    dimensions=[16, 32],
    group_sizes=[1, 2, 4, 8],
    radii=[0.5, 2.0],
    seeds=[21101, 21102, 21103, 21104, 21105],
    initials=512,
    rank_tolerance=1e-10,
    note='Geometry only. Targets, active faces and ranks come from minimal_teaching.',
)


def initials_cpu(n, d, seed):
    """The frozen initial-state law of minimal_teaching.initials, evaluated on the CPU."""
    gen = torch.Generator().manual_seed(seed)
    rare = torch.rand(n, generator=gen) < .1
    x = 2 * torch.rand(n, d, generator=gen) - 1
    x[rare] *= .1
    x[~rare, 0] *= .1
    sign = torch.where(torch.rand(n, generator=gen) < .5, -1., 1.)
    mag = 6 + 4 * torch.rand(n, generator=gen)
    x[rare, 0] = (sign * mag)[rare]
    return x.double()


def basis_ganged(d, g):
    """One channel per contiguous group of g actuators; g = 1 is one channel per actuator."""
    assert d % g == 0
    m = d // g
    P = torch.zeros(d, m, dtype=torch.float64)
    for j in range(m):
        P[j * g:(j + 1) * g, j] = 1.0 / g ** .5
    return P


def basis_mirror(d, m):
    """Even Fourier modes about node 0: rows i and d-i of P coincide."""
    cols, q = [], 0
    while len(cols) < m and q <= d:
        v = torch.cos(2 * torch.pi * q * torch.arange(d, dtype=torch.float64) / d)
        if v.norm() > 1e-12:
            cols.append(v / v.norm())
        q += 1
    return torch.stack(cols[:m], dim=1)


def basis_random(d, m, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, m, generator=gen, dtype=torch.float64), mode='reduced').Q


def tally(P, radius, seed, n_initials):
    """Query counts and the spectral gap for one basis, radius and seed."""
    d, m = P.shape
    assert torch.allclose(P.T @ P, torch.eye(m, dtype=P.dtype), atol=1e-12), 'P must be orthonormal'
    x = initials_cpu(n_initials, d, seed)
    v = mt.Oracle(mt.states(x), radius, CFG['shape']).target
    geo = mt.geometry(v, P, radius, CFG['shape'])
    k = geo['face_dim']
    r = geo['rank']
    hybrid = torch.minimum(k, torch.full_like(k, m))
    active = k > 0
    # Spectral gap of the same Gram matrix the frozen rank test uses.
    L = P.T[None, :, :] * geo['active'][:, None, :]
    val = torch.linalg.eigvalsh(L @ L.transpose(1, 2)).flip(-1)
    order = torch.arange(m)[None, :]
    discarded = torch.where(order >= r[:, None], val, torch.full_like(val, -1.)).amax()
    retained = torch.where(order < r[:, None], val, torch.full_like(val, float('inf'))).amin()
    states = len(v)
    return dict(
        states=states, active_states=int(active.sum()),
        mean_k=float(k.double().mean()), mean_rank=float(r.double().mean()),
        mean_hybrid=float(hybrid.double().mean()),
        mean_student_basis=float((active.long() * m).double().mean()),
        mean_face=float(k.double().mean()),
        strict_saving_states=int((r < hybrid).sum()),
        total_rank=int(r.sum()), total_hybrid=int(hybrid.sum()),
        largest_discarded_eigenvalue=max(float(discarded), 0.0),
        smallest_retained_eigenvalue=None if not torch.isfinite(retained) else float(retained),
    )


def main():
    torch.set_num_threads(1)
    rows = []
    for d in CFG['dimensions']:
        for g in CFG['group_sizes']:
            m = d // g
            for radius in CFG['radii']:
                for seed in CFG['seeds']:
                    for family, P in [('ganged', basis_ganged(d, g)),
                                      ('mirror', basis_mirror(d, m)),
                                      ('random', basis_random(d, m, seed + 311))]:
                        if family != 'ganged' and g == 1:
                            continue      # m = d makes every basis a full orthonormal frame
                        rec = tally(P, radius, seed, CFG['initials'])
                        rows.append(dict(dimension=d, group=g, student_dimension=m,
                                         radius=radius, seed=seed, family=family, **rec))
    report = dict(
        passed=True, config=CFG, rows=rows,
        source_sha256=mt.digest(__file__),
        interface_sha256=mt.digest(mt.__file__),
    )
    mt.write(OUT / 'report.json', report)
    lock = dict(config=CFG, source_sha256=report['source_sha256'],
                interface_sha256=report['interface_sha256'],
                report_sha256=mt.digest(OUT / 'report.json'))
    mt.write(OUT / 'protocol_lock.json', lock)
    summary = {}
    for family in ['ganged', 'mirror', 'random']:
        sub = [r for r in rows if r['family'] == family]
        if not sub:
            continue
        summary[family] = dict(
            conditions=len(sub),
            saving_vs_hybrid=1 - sum(r['total_rank'] for r in sub) / max(sum(r['total_hybrid'] for r in sub), 1),
            strict_fraction=sum(r['strict_saving_states'] for r in sub) / max(sum(r['active_states'] for r in sub), 1),
            largest_discarded_eigenvalue=max(r['largest_discarded_eigenvalue'] for r in sub),
        )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
