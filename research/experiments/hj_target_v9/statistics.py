"""Predeclared paired arithmetic tests; descriptive budget threshold curves."""
import math
import numpy as np
from scipy.stats import t as student_t


def paired(candidate,reference,margin=1.,alpha=.05/3):
    a=np.asarray(candidate,dtype=float);b=np.asarray(reference,dtype=float)
    assert a.ndim==b.ndim==1 and len(a)==len(b)>=2
    assert np.isfinite(a).all() and np.isfinite(b).all() and (b>0).all()
    d=a-margin*b;n=len(d);mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(n))
    upper=mean+float(student_t.ppf(1-alpha,n-1))*se
    p=float(student_t.cdf(mean/se,n-1)) if se else (0. if mean<0 else 1.)
    return dict(n=n,margin=margin,alpha=alpha,difference_mean=mean,standard_error=se,
        upper_difference=upper,p_one_sided=p,ratio=float(a.mean()/b.mean()),
        paired_seed_ratios=(a/b).tolist(),negative_seed_differences=int((d<0).sum()))


def family_decision(means,candidate='hjb_target',references=('short','dpc','warm')):
    results={}
    for name in references:
        superiority=paired(means[candidate],means[name])
        noninferiority=paired(means[candidate],means[name],margin=1.01)
        results[name]=dict(superiority=superiority,noninferiority=noninferiority)
    return dict(references=results,
        all_reference_superiority_with_observed_1pct=all(v['superiority']['upper_difference']<0 and v['superiority']['ratio']<=.99 for v in results.values()),
        all_reference_1pct_noninferiority=all(v['noninferiority']['upper_difference']<=0 for v in results.values()))
