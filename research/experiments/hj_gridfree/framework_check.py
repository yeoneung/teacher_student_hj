"""Independent analytic checks for the HJ exposition; no primary data is modified.

These small, deterministic calculations are CPU quadrature/algebra checks, not
additional learning benchmarks or evidence of nonlinear algorithmic superiority.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def integral(fn):
    return quad(fn, 0., 1., epsabs=1e-12, epsrel=1e-12)[0]


def state(a, t):
    return np.exp(-a * t)


def closed_cost(a):
    stage = .5 * (1. + a * a)
    return stage * (-np.expm1(-2. * a) / (2. * a) if a else 1.) + .625 * np.exp(-2. * a)


def hamiltonian(x, p, u):
    return .5 * (x * x + u * u) + p * u


def main():
    out = Path('experiments/results/hj_gridfree_framework_v2')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in [0., .5, 1., 1.25, 2., 3., 4.]:
        direct = integral(lambda t: .5 * (1. + a*a) * state(a, t)**2) + .625 * state(a, 1.)**2
        regret = integral(lambda t: .5 * (a-1.25)**2 * state(a, t)**2)
        opportunity = integral(lambda t: .5 * .75**2 * state(a, t)**2)
        gap = direct - closed_cost(2.)
        # W = V^b + .17(1+t)x^2 + .03t, including nonzero terminal error.
        def residual(t, x):
            return .17*x*x + .03 - 4.*.17*(1.+t)*x*x
        def advantage_w(t):
            x = state(a, t)
            p = (1.25 + 2.*.17*(1.+t))*x
            return hamiltonian(x, p, -a*x) - hamiltonian(x, p, -2.*x)
        approx_adv = integral(advantage_w)
        terminal_error = lambda x: .34*x*x + .03
        correction = integral(lambda t: residual(t, state(a, t)) - residual(t, state(2., t)))
        correction += -terminal_error(state(a, 1.)) + terminal_error(state(2., 1.))
        rows.append(dict(gain=a, direct_cost=direct, closed_form_cost=closed_cost(a),
                         gap=gap, integrated_regret=regret, integrated_opportunity=opportunity,
                         identity_error=abs(gap-(regret-opportunity)),
                         approximate_advantage=approx_adv, evaluation_correction=correction,
                         approximate_identity_error=abs(gap-(approx_adv+correction))))
    errors = [r['identity_error'] for r in rows] + [r['approximate_identity_error'] for r in rows]
    errors += [abs(r['direct_cost']-r['closed_form_cost']) for r in rows]
    assert max(errors) < 2e-12, errors

    # A constrained minimizer has an additional nonnegative boundary term.
    rng = np.random.default_rng(137)
    diag_r = np.array([2., 5.])
    linear = np.array([3., -4.])
    u_plus = np.clip(-linear / diag_r, -.6, .6)
    candidates = rng.uniform(-.6, .6, (1000, 2))
    diff = candidates - u_plus
    regret_direct = .5*np.sum(candidates*candidates*diag_r, axis=1) + candidates @ linear
    regret_direct -= .5*np.sum(u_plus*u_plus*diag_r) + u_plus @ linear
    quadratic = .5*np.sum(diff*diff*diag_r, axis=1)
    boundary = diff @ (diag_r*u_plus + linear)
    constrained_error = float(np.max(np.abs(regret_direct - quadratic - boundary)))
    assert constrained_error < 2e-12 and np.min(boundary) >= -1e-12

    # Explicit invertible state transformation; no metric assumptions on V.
    transform = np.array([[2., .5], [-.3, 1.2]])
    velocity = np.array([.7, -.2])
    covector = np.array([1.1, .9])
    transformed_covector = np.linalg.solve(transform.T, covector)
    coordinate_error = float(abs(transformed_covector @ (transform @ velocity) - covector @ velocity))
    assert coordinate_error < 1e-12

    result = dict(scope='Analytic exposition checks only; frozen primary experiment unchanged',
                  continuous_rows=rows, max_continuous_or_quadrature_error=max(errors),
                  constrained_regret_max_error=constrained_error,
                  constrained_boundary_min=float(np.min(boundary)),
                  coordinate_pairing_error=coordinate_error,
                  action_metric_example=dict(squared_euclidean=[1., .04], hamiltonian_regret=[.5, 2.]))
    (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')

    gains = np.linspace(0., 3., 241)
    costs = np.array([closed_cost(a) for a in gains])
    occupation = np.array([-np.expm1(-2*a)/(2*a) if a else 1. for a in gains])
    regret = .5*(gains-1.25)**2*occupation
    opportunity = .5*.75**2*occupation
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3), layout='constrained')
    ax[0].plot(gains, costs, color='#1b6578', lw=2, label='Student cost')
    ax[0].axhline(.625, color='#696969', ls='--', lw=1)
    ax[0].scatter([2., 1.25], [closed_cost(2.), closed_cost(1.25)], color=['#555555', '#bd4e35'], zorder=3)
    ax[0].annotate('Copies teacher', (2., .625), xytext=(1.25, .81),
                   arrowprops={'arrowstyle':'-', 'color':'#666666'}, fontsize=9)
    ax[0].annotate('Hamiltonian improvement', (1.25, closed_cost(1.25)), xytext=(.38, .41),
                   arrowprops={'arrowstyle':'-', 'color':'#bd4e35'}, fontsize=9)
    ax[0].set(xlabel=r'Student gain $a$ in $u=-ax$', ylabel=r'Cost $J^{\pi_a}(1)$',
              title='Action copying is not the cost target', ylim=(.37, 1.16))
    ax[1].plot(gains, regret, label=r'Integrated student regret $r$', color='#c37824')
    ax[1].plot(gains, opportunity, label=r'Integrated teacher opportunity $\delta$', color='#1b6578')
    ax[1].plot(gains, regret-opportunity, label=r'$r-\delta$', color='#bd4e35', lw=2)
    ax[1].scatter(gains[::16], (costs-.625)[::16], marker='x', color='black', s=23,
                  label='Independent cost difference', zorder=3)
    ax[1].axhline(0., color='#999999', lw=.7)
    ax[1].set(xlabel=r'Student gain $a$', ylabel='Integrated cost contribution',
              title='Performance identity on student trajectories')
    ax[1].legend(fontsize=8, loc='upper right')
    for axis in ax: axis.grid(alpha=.15)
    for ext in ['pdf', 'png']:
        fig.savefig(Path('paper/figures')/f'hj_gridfree_framework.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k != 'continuous_rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
