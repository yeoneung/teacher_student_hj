"""Exact rational illustrations separating teaching approximations."""
from fractions import Fraction as Q
from experiments.hj_gridfree.engine import write


def rollout(theta,a):
    x=Q(1);cost=Q(0)
    for coefficient in [Q(1),a]:
        u=coefficient*theta;cost+=u*u/Q(10);x+=u
    return cost+x*x


def surrogate(theta,phi,a):
    x1=1+phi;x2=x1+a*phi;p1=2*x2
    first=(theta*theta-phi*phi)/10+p1*(theta-phi)
    last=a*a*(theta*theta-phi*phi)/10+(x1+a*theta)**2-x2*x2
    return rollout(phi,a)+first+last


def main():
    points=[Q(-3,2),Q(-1),Q(-1,3),Q(0),Q(1,5),Q(2,3),Q(1)];checks=0
    for a in [Q(-2),Q(1)]:
        for theta in points:
            expected=1+2*(1+a)*theta+((1+a)**2+(1+a*a)/10)*theta*theta
            assert rollout(theta,a)==expected;checks+=1
            for phi in points:
                assert surrogate(theta,phi,a)-rollout(theta,a)==-(1+2*a)*(theta-phi)**2;checks+=1
    assert rollout(Q(-5,3),Q(1))==6
    assert surrogate(Q(-5,3),Q(0),Q(1))==Q(-7,3)
    assert rollout(Q(-10,21),Q(1))==Q(1,21)
    assert rollout(Q(2,3),Q(-2))==Q(1,3)
    assert ((4-(-2))**2+(-8-(-2))**2)/2==36
    phi=Q(0);iterations=[]
    for k in range(8):
        iterations.append(dict(iteration=k,teacher=str(phi),true_cost=str(rollout(phi,Q(-2)))))
        new=Q(2,9)+Q(2,3)*phi
        assert rollout(new,Q(-2))<=rollout(phi,Q(-2));phi=new
    record=dict(passed=True,exact_polynomial_checks=checks,
        model='x_next=x+u, H=2, x0=1, c=u^2/10, g=x^2, u0=theta, u1=a*theta',
        a_minus_two=dict(full_cost='1-2theta+3theta^2/2',prefix_cost='1+2theta+11theta^2/10',
            full_derivative_at_zero=-2,prefix_derivative_at_zero=2,surrogate_minus_true='3(theta-phi)^2',
            uniform_single_pair_gradients=[4,-8],gradient_mean=-2,gradient_variance=36,corrected_anchor_variance=0,
            exact_surrogate_minimization_iterations=iterations),
        a_plus_one=dict(full_cost='1+4theta+21theta^2/5',anchor_surrogate='1+4theta+6theta^2/5',
            shared_anchor_derivative=4,surrogate_minimizer='-5/3',true_cost_at_surrogate_minimizer=6,
            incumbent_cost=1,surrogate_minimum='-7/3'),
        scope='Constructed exact algebraic illustrations, not selected control benchmarks, primary performance or a numerical HJB grid.')
    write('experiments/results/hj_adaptive_dev_v7e/analytic_examples.json',record)
    print('EXACT RATIONAL TEACHER-INTERFACE EXAMPLES VERIFIED')


if __name__=='__main__':main()
