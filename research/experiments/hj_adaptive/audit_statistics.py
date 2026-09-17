"""Check arithmetic mean-cost decisions against SciPy's independent t-test API."""
import argparse
import numpy as np
from scipy.stats import ttest_1samp
from experiments.hj_gridfree.engine import write
from .report import comparison


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    b=np.linspace(2.,4.,10)[:,None]*np.linspace(.7,1.3,16)[None,:];rows=[]
    for factor in [.8,1.,1.3]:
        a=factor*b;r=comparison(a,b,draws=20);d=a.mean(1)-1.01*b.mean(1)
        reference=ttest_1samp(d,0.,alternative='less')
        upper=float(reference.confidence_interval(confidence_level=1-.05/3).high)
        assert abs(upper-r['noninferiority_upper_mean_difference'])<1e-12
        assert abs(reference.pvalue-r['noninferiority_pvalue'])<1e-12
        assert r['noninferior']==(factor<=1.01) and r['superior_one_percent']==(factor<=.99)
        rows.append(dict(factor=factor,upper_mean_difference=upper,pvalue=float(reference.pvalue),noninferior=r['noninferior']))
    b=np.array([1.]*9+[100.])[:,None];a=b*np.array([.5]*9+[1.1])[:,None]
    counterexample=comparison(a,b,draws=20)
    assert counterexample['corrected_upper_ratio']<counterexample['pooled_ratio']
    write(args.output,dict(passed=True,rows=rows,estimand_counterexample=counterexample,
        scope='Synthetic validation of the statistical implementation and the distinction between geometric and arithmetic targets. These are not primary control outcomes.'))
    print('ARITHMETIC COST-DIFFERENCE STATISTICAL AUDIT PASSED')


if __name__=='__main__':main()
