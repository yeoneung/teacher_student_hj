"""Check the HJB connection on a solvable scalar problem, separately from v1/v2."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad


def pstar(t):
    z = np.exp(2*(t-1))/9.
    return (1+z)/(1-z)


def dpstar(t):
    z = np.exp(2*(t-1))/9.
    return 4*z/(1-z)**2


def optimal_state(t):
    z = np.exp(2*(t-1))/9.
    z0 = np.exp(-2)/9.
    return np.exp(-t)*(1-z)/(1-z0)


def quad01(fn):
    return quad(fn, 0., 1., epsabs=1e-12, epsrel=1e-12)[0]


def direct_cost(a):
    occupancy = -np.expm1(-2*a)/(2*a) if a else 1.
    return .5*(1+a*a)*occupancy + .625*np.exp(-2*a)


def main():
    # W(t,x) = P(t)*x^2/2 + C(t). P* solves P'=P^2-1, P(1)=5/4.
    fields = {
        'teacher': (lambda t: 1.25, lambda t: 0., lambda t: 0., lambda t: 0.),
        'optimal': (pstar, dpstar, lambda t: 0., lambda t: 0.),
        'approximate': (lambda t: 1.25+.34*(1+t), lambda t: .34,
                        lambda t: .03*t, lambda t: .03),
    }
    rows = []
    for name, (p, dp, c, dc) in fields.items():
        for a in [0., .5, 1., 1.25, 2., 3., 4.]:
            residual = quad01(lambda t: .5*(dp(t)+1-p(t)**2)*np.exp(-2*a*t)+dc(t))
            regret = quad01(lambda t: .5*(p(t)-a)**2*np.exp(-2*a*t))
            terminal_correction = (.625-.5*p(1.))*np.exp(-2*a)-c(1.)
            direct_gap = direct_cost(a)-(.5*p(0.)+c(0.))
            rows.append(dict(field=name, gain=a, direct_gap=direct_gap,
                             integrated_hjb_residual=residual, integrated_action_regret=regret,
                             terminal_correction=terminal_correction,
                             master_identity_error=abs(direct_gap-residual-regret-terminal_correction)))
    max_error = max(r['master_identity_error'] for r in rows)
    optimal_direct = quad01(lambda t: .5*(1+pstar(t)**2)*optimal_state(t)**2)
    optimal_direct += .625*optimal_state(1.)**2
    optimal_verification_error = abs(optimal_direct-.5*pstar(0.))
    hjb_zero_error = max(abs(.5*(dpstar(t)+1-pstar(t)**2)) for t in np.linspace(0., 1., 51))
    assert max(max_error, optimal_verification_error, hjb_zero_error) < 2e-12
    assert direct_cost(2.) > direct_cost(1.25) > optimal_direct

    # Independently minimize the Euler-step quadratic action return analytically.
    # This tests the normalized Bellman residual -> HJB residual connection.
    consistency = {}
    t, x = .2, .8
    for name, (p, dp, c, dc) in fields.items():
        exact_residual = .5*(dp(t)+1-p(t)**2)*x*x+dc(t)
        entries = []
        for h in [.1, .05, .025, .0125, .00625, .003125]:
            pn = p(t+h)
            minimizing_u = -pn*x/(1+h*pn)
            next_x = x+h*minimizing_u
            action_return = .5*h*(x*x+minimizing_u**2)+.5*pn*next_x**2+c(t+h)
            normalized = (action_return-(.5*p(t)*x*x+c(t)))/h
            entries.append(dict(step=h, normalized_bellman_residual=normalized,
                                hjb_residual=exact_residual, error=abs(normalized-exact_residual)))
        ratios = [entries[i]['error']/entries[i+1]['error'] for i in range(len(entries)-1)]
        assert all(1.7 < ratio < 2.3 for ratio in ratios), (name, ratios)
        consistency[name] = dict(rows=entries, error_reduction_ratios=ratios)

    # Reevaluation of the greedy student changes the value field: it is not V^b.
    gain = 1.25
    equilibrium_p = (1+gain*gain)/(2*gain)
    p_improved = lambda t: equilibrium_p+(1.25-equilibrium_p)*np.exp(2*gain*(t-1))
    dp_improved = lambda t: 2*gain*(p_improved(t)-equilibrium_p)
    reevaluation_error = max(abs(.5*(dp_improved(t)+1-p_improved(t)**2)
                                +.5*(gain-p_improved(t))**2) for t in np.linspace(0., 1., 51))
    assert reevaluation_error < 2e-12
    result = dict(scope='Analytic HJB exposition check; no nonlinear policies trained or modified',
                  costs=dict(teacher=direct_cost(2.), greedy_improvement=direct_cost(1.25), optimal=optimal_direct),
                  max_master_identity_error=max_error,
                  optimal_policy_verification_error=optimal_verification_error,
                  optimal_field_hjb_residual_max_abs=hjb_zero_error,
                  reevaluated_student_residual_identity_error=reevaluation_error,
                  master_identity_rows=rows, bellman_hjb_consistency=consistency)
    out = Path('experiments/results/hj_gridfree_hjb_v3')
    out.mkdir(parents=True, exist_ok=True)
    (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['master_identity_rows', 'bellman_hjb_consistency']}, indent=2))


if __name__ == '__main__':
    main()
