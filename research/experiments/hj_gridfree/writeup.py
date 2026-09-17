"""Generate paper sections and a Korean research report from the completed summary."""
import json
import re
from pathlib import Path
import numpy as np
from .report import tex_table


def polish_tex(text):
    saved=[]
    def protect(match):
        saved.append(match.group(0)); return chr(0xE000+len(saved)-1)
    text=re.sub(r'\\(?:cite|ref|eqref|label|input|includegraphics|bibliography|bibliographystyle|url|href|texttt)(?:\[[^\]]*\])?\{[^}]*\}',protect,text)
    text=re.sub(r'(?<=[A-Za-z])(?=\d)',' ',text)
    text=re.sub(r'(?<=\d),(?=\d)',', ',text)
    text=re.sub(r'(?<=[A-Za-z])(?=\[-?\d)',' ',text)
    text=re.sub(r'(?<=[;:])(?=\S)',' ',text)
    text=text.replace('20480000','20\\,480\\,000').replace('8192000','8\\,192\\,000')
    for i,item in enumerate(saved): text=text.replace(chr(0xE000+i),item)
    return text


def main():
    root=Path('experiments/results/hj_gridfree_primary_v1')
    s=json.loads((root/'report/summary.json').read_text(encoding='utf-8'))
    cases=s['cases']; sections=Path('paper/sections'); sections.mkdir(exist_ok=True)
    def nominal(f): return [cases[f'{f}_d{d}_nominal'] for d in [32,64,128,256]]
    def range_text(values):
        values=[0. if abs(v)<.005 else v for v in values]
        return f'[{min(values):.2f}, {max(values):.2f}]'
    savings={f:100*(1-s['training'][f]['block16']['total_seconds']/s['training'][f]['dpc']['total_seconds']) for f in ['mechanical','reaction']}
    mr=range_text([c['comparisons']['best_classical']['reduction_percent'] for c in nominal('mechanical')])
    rr=range_text([c['comparisons']['best_classical']['reduction_percent'] for c in nominal('reaction')])
    mn=range_text([c['native_subset']['hybrid_comparison']['reduction_percent'] for c in nominal('mechanical')])
    rn=range_text([c['native_subset']['hybrid_comparison']['reduction_percent'] for c in nominal('reaction')])
    abstract=(
        'We formulate teacher--student control through Hamilton--Jacobi--Bellman (HJB) policy evaluation and improvement. '
        'The teacher supplies a value field; the student approximates the Hamiltonian minimization step of HJB, whose exact policy-iteration fixed point satisfies optimality. '
        'Under smooth deterministic assumptions, the performance difference equals accumulated student Hamiltonian regret minus the teacher improvement opportunity, which is the negative HJB residual of the teacher value. '
        'A general residual identity connects this comparison to HJB verification and distinguishes action-selection error from value-evaluation error. '
        'A finite-step Bellman continuation carries the same interpretation into a grid-free learning objective: a short neural prefix receives its boundary value and state sensitivity from an exactly simulated teacher tail. '
        'We provide a discrete block identity and an optional execution gate with a conditional relative-cost guarantee. '
        'The computational study comprises50 neural fits and6144 independent initial states on coupled mechanical and reaction--diffusion rings, transferring weights trained at dimension32 up to256. '
        f'Against a native CPU/GPU hybrid on matched nominal subsets, signed cost reductions are{mn}\\% and{rn}\\%, respectively; negative values indicate degradation. '
        f'Total training-related time is{savings["mechanical"]:.1f}\\% and{savings["reaction"]:.1f}\\% lower than full differentiable policy-cost training, with comparable or somewhat higher nominal cost. '
        'The theory supplies a unified interpretation grounded in established policy evaluation and improvement; the experiments measure the benefits and limits of one temporal implementation.\n')
    (sections/'hj_gridfree_abstract.tex').write_text(polish_tex(abstract),encoding='utf-8')
    lines=[]
    lines.append('\\subsection{Transfer with fixed neural weights}\nTable~\\ref{tab:gridfree_nominal} reports independent nominal costs. Neural entries average the five trained policies on the same512 states; no larger-size neural refitting is used. The GPU classical entry is a casewise best feasible search result, not a global optimum.\n')
    rows=[]
    for fam in ['mechanical','reaction']:
        for c in nominal(fam):
            rows.append([fam[:4],c['dimension']]+[f'{c["means"][m]:.4f}' for m in ['lqr','best_classical','block16','dpc','truncated16']])
    lines.append('\\begin{table}[htbp]\\centering\\small\n'+tex_table(['Family','$d$','LQR','GPU search','Bellman16','DPC','Trunc16'],rows,'lrrrrrr')+'\\caption{Nominal mean costs on512 independent states per row.}\\label{tab:gridfree_nominal}\\end{table}\n')
    lines.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{figures/hj_gridfree_scaling.pdf}\\caption{Size transfer under unchanged neural weights. The model and shared input constraint remain coupled at every size.}\\end{figure}\n')
    lines.append(f'The nominal reduction relative to the GPU search is{mr}\\% for the mechanical systems and{rr}\\% for the reaction systems. This comparison alone does not establish superiority over strong CPU optimization. Table~\\ref{{tab:gridfree_native}} therefore uses the prespecified matched native subsets.\n')
    rows=[]
    for fam in ['mechanical','reaction']:
        for c in nominal(fam):
            v=c['native_subset']; z=v['hybrid_comparison']; ci=z['ci95_percent']
            rows.append([fam[:4],c['dimension'],f'{v["bellman_mean"]:.4f}',f'{v["two_start_mean"]:.4f}',f'{v["hybrid_mean"]:.4f}',f'{z["reduction_percent"]:.2f}',f'[{ci[0]:.2f},{ci[1]:.2f}]'])
    lines.append('\\begin{table}[htbp]\\centering\\scriptsize\n'+tex_table(['Family','$d$','Bellman16','CPU2start','CPU/GPU','Reduction(\\%)','95\\% CI'],rows,'lr rrrrr')+'\\caption{Matched first128 nominal states. CPU/GPU is native1024-iteration optimization from LQR, zero, and the best full GPU search plan. Negative reduction favors the hybrid. Intervals resample seeds and states.}\\label{tab:gridfree_native}\\end{table}\n')
    lines.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{figures/hj_gridfree_native.pdf}\\caption{Paired nominal cost reductions against native comparators. Error bars are descriptive crossed bootstrap95\\% intervals.}\\end{figure}\n')
    lines.append('\\subsection{The teacher continuation versus other learning objectives}\n')
    for fam in ['mechanical','reaction']:
        nc=nominal(fam); dc=[c['comparisons']['dpc']['reduction_percent'] for c in nc]; tc=[c['comparisons']['truncated16']['reduction_percent'] for c in nc]
        lines.append(f'For {fam} systems, the signed nominal cost reduction of Bellman16 relative to DPC ranges from{range_text(dc)}\\%; relative to the same short prefix without a teacher continuation, it ranges from{range_text(tc)}\\%. These are distinct contrasts: improving on action imitation does not prove an improvement over full policy-cost training.\n')
    lines.append('\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{figures/hj_gridfree_learning.pdf}\\caption{Validation quality versus measured fit, setup, and preparation time. Shading is one standard deviation across the five seeds. Validation execution time is excluded from the horizontal axis but included in Table~\\ref{tab:gridfree_training}.}\\end{figure}\n')
    rows=[]
    for fam in ['mechanical','reaction']:
        for mode in s['configuration']['modes']:
            t=s['training'][fam][mode]; mem=s['isolated_memory'][f'{fam}_{mode}']['incremental_peak_bytes']/2**20
            rows.append([fam[:4],mode,f'{t["fit_seconds"]:.2f}',f'{t["preparation_seconds"]:.2f}',f'{t["total_seconds"]:.2f}',f'{mem:.1f}'])
    lines.append('\\begin{table}[htbp]\\centering\\small\n'+tex_table(['Family','Loss','Fit(s)','Prep(s)','Total(s)','Graph(MiB)'],rows,'llrrrr')+'\\caption{Mean training cost per seed. Total includes setup and validation. Graph memory is measured in a fresh process and includes weights and optimizer, excluding dataset and CUDA driver allocations.}\\label{tab:gridfree_training}\\end{table}\n')
    lines.append(f'Bellman16 reduces measured total training cost by{savings["mechanical"]:.2f}\\% and{savings["reaction"]:.2f}\\% relative to DPC in the two families. Both simulate20480000 model transitions per fit in this implementation, including masked inactive steps; neural evaluations fall from20480000 to8192000. Thus the speed reduction comes from the neural computational graph, not from claiming that teacher simulation is free.\n')
    lines.append('\\subsection{Distribution shifts and the optional gate}\n')
    rows=[]
    for fam in ['mechanical','reaction']:
        for d in [32,64,128,256]:
            c=cases[f'{fam}_d{d}_shift']; rows.append([fam[:4],d]+[f'{c["means"][m]:.4f}' for m in ['best_classical','block16','dpc','truncated16']])
    lines.append('\\begin{table}[htbp]\\centering\\small\n'+tex_table(['Family','$d$','GPU search','Bellman16','DPC','Trunc16'],rows,'lrrrrr')+'\\caption{Combined coupling, budget, and initial-condition shift;256 states per row.}\\end{table}\n')
    gates=[c['gate'] for c in cases.values()]
    lines.append(f'The exact block gate selected the student in{100*min(g["student_block_fraction"] for g in gates):.1f}--{100*max(g["student_block_fraction"] for g in gates):.1f}\\% of evaluated blocks. The maximum observed cost increase against the base was{max(g["max_cost_increase_vs_base"] for g in gates):.2g}. This checks the implementation of the relative-cost property on the evaluated states; it does not create a state-safety guarantee. Gate deployment uses additional full teacher continuations.\n')
    lines.append('\\subsection{Deployment latency}\n')
    tailrows=[]
    for fam in ['mechanical','reaction']:
        for d in [32,256]:
            for dist in ['nominal','shift']:
                c=cases[f'{fam}_d{d}_{dist}']; native=c['native_subset']['hybrid_comparison']
                tailrows.append([fam[:4],d,dist,f'{c["tails"]["block16"]["p95_cost"]:.3f}',f'{c["tails"]["dpc"]["p95_cost"]:.3f}',f'{native["paired_ratio_p95"]:.3f}'])
    lines.append('\\begin{table}[htbp]\\centering\\small\n'+tex_table(['Family','$d$','Condition','Bellman $q_{.95}$','DPC $q_{.95}$','Ratio $q_{.95}$'],tailrows,'lr lrrr')+'\\caption{Tail diagnostics at the training and largest transferred size. Cost quantiles pool the five seeds and all test states; the paired Bellman/native-hybrid ratio uses the prescribed native subset. These are empirical percentiles, not independent-sample confidence bounds.}\\end{table}\n')
    rows=[]
    for fam in ['mechanical','reaction']:
        for d in [32,64,128,256]:
            t=s['timing'][f'{fam}_d{d}']['methods']; rows.append([fam[:4],d,f'{1000*t["block16_action"]["wall_median_ms"]:.1f}',f'{t["block16_full_plan"]["wall_median_ms"]:.3f}',f'{t["native_two_start1024"]["wall_median_ms"]:.1f}',f'{t["block16_gate_full_plan"]["wall_median_ms"]:.3f}'])
    lines.append('\\begin{table}[htbp]\\centering\\small\n'+tex_table(['Family','$d$','Action($\\mu$s)','Plan(ms)','CPU2(ms)','Gate(ms)'],rows,'lrrrrr')+'\\caption{Synchronized neural GPU timings and serial native two-start1024-iteration timing. Plan and gate include all40 steps. CPU2 quality corresponds to the CPU2start column on matched subsets; it is not the stronger CPU/GPU hybrid.}\\end{table}\n')
    failures=sum(c['targets']['hj_quality_advantage_at_least_5_over_both_mse_and_dpc'] or c['targets']['dpc_quality_within_1_at_half_total_training_time'] for c in cases.values())
    lines.append(f'The predeclared strict HJ-specific target is met in{failures} of16 case configurations: at least5\\% lower cost than both MSE and DPC, or DPC quality within1\\% using at most half its total training cost. Smaller measured tradeoffs are reported rather than substituted retrospectively for that target. The largest independent scaled cost replay discrepancy was{max(c["audit"]["max_independent_scaled_cost_error"] for c in cases.values()):.3g}; the largest numerical input-budget ratio was{max(c["audit"]["max_budget_ratio"] for c in cases.values()):.9f}.\n')
    (sections/'hj_gridfree_results.tex').write_text(polish_tex('\n'.join(lines)),encoding='utf-8')
    discussion=r'''The conceptual result is a common HJB language for teacher--student control. A teacher supplies a future-cost field, its gradient prices changes in state velocity, and the student's Hamiltonian advantage accumulates into a closed-loop performance difference. Equation~\eqref{eq:hjb-master} connects this comparison directly to HJB verification. For the teacher field, its HJB residual equals the negative improvement opportunity; for an optimal field, the residual vanishes and total suboptimality is accumulated action-selection regret. Equation~\eqref{eq:regret-defect} therefore explains how a student can improve on an accurately evaluated but suboptimal teacher. The discrete block counterpart preserves the performance comparison without a smooth value function.

The algorithm occupies a specific place in HJB policy iteration. It approximates finite-step policy improvement against one fixed teacher continuation. It does not repeatedly reevaluate each learned student as the next teacher, and minimizing the actor loss does not change the fixed teacher's HJB residual. The distinction matters: a useful one-step improvement, a policy-iteration convergence result, and a certified HJB solution require different evidence. The conditional bound in Corollary~\ref{cor:hjb-gap} makes the uniform residual premise for an optimality certificate explicit; that premise has not been established on the nonlinear benchmarks.

The finite-step implementation follows from that interpretation. Teacher evaluation supplies the boundary condition beyond the student prefix, while the boundary derivative in~\eqref{eq:boundary-gradient} transmits its consequences into learning. The tested prefix changes where control responsibility is assigned during training, while retaining the model horizon. This distinguishes a teacher continuation from simply stopping the rollout early. The experiments separate amortizing trajectory search, using consequence-sensitive objectives, and choosing a temporal teacher boundary. Improvement from the first two does not establish an additional benefit from the third.

The simulated continuation is the actual return of the fixed teacher in the discrete model, even when that teacher controls poorly. Poor teacher behavior is different from inaccurate teacher-value approximation. The former changes the improvement opportunity and training landscape; the latter introduces the additional error in~\eqref{eq:approximate-field}. Fixed teacher-trajectory training also does not guarantee coverage of learned-policy states. The shift results are consistent with this coverage concern, but they do not isolate its cause from changes in dynamics or constraints. We have not measured the global block infima or separately estimated the regret and opportunity terms on the nonlinear benchmarks.

The strongest classical comparison includes native optimization initialized by all preceding GPU searches. Its cost must be reported on the same subset, and its latency cannot be replaced by the latency of a cheaper two-start method. Conversely, neural deployment latency excludes training only when training is explicitly treated as an amortized preparation cost. The applicable use case is repeated control under related models and distributions, not one isolated initial condition with free preparation.

Several limitations remain. Both families are homogeneous rings with known exact dynamics and a common shared input ball. Increasing size enlarges the ring; no arbitrary-graph, heterogeneous-actuator, fixed-domain PDE-refinement, real-plant, or noisy-feedback claim follows. The discrete horizon and parameters are fixed rather than refined toward a continuous-time optimum. A casewise best returned plan is an empirical feasible envelope, not a global optimality certificate. The native solver's radial parameterization can affect conditioning and local convergence. The comparison concerns independent finite-horizon initial-value plans; warm-started receding-horizon optimization under disturbances is not benchmarked. Five training seeds give limited precision, and the multiple descriptive intervals are not family-wise corrected.

The theoretical contribution is a synthesis and explicit derivation for teacher--student learning, grounded in established rollout, cost-sensitive imitation, and HJ policy evaluation. Hamiltonian-guided imitation predates this study. The claimed contribution is the connected interpretation of teacher suboptimality, student error, constraint geometry, value-evaluation error, and temporal boundary design, together with an auditable implementation. It does not establish a new general policy-iteration theorem or resolve HJB sample complexity. Learned tail-value approximation, adaptive query allocation, and on-policy teacher refresh remain untested extensions.

The conceptual claim and the empirical claim should be assessed separately. The former is that teacher--student control can be understood as transferring an evaluation of future consequences and learning to improve the induced Hamiltonian choices. The latter is that one grid-free Bellman implementation offers a measured computation/quality tradeoff on the tested coupled systems. The predeclared strict algorithmic target was not met; a strengthened interpretation does not change that outcome. Conversely, that outcome does not invalidate the performance identities or the explanatory role of the framework. Establishing a stronger methodological novelty would require additional results beyond the present synthesis and experiments.
'''
    (sections/'hj_gridfree_discussion.tex').write_text(discussion,encoding='utf-8')
    # Hand-written mathematical source is not passed through numeric prose cleanup.
    # In particular, that cleanup must not alter font encodings or math commands.
    # A concise Korean entry point retains the negative and shifted comparisons.
    ko=['# 격자 없는 HJ teacher–student: 독립 실험 결과','',
        'PMP 라벨이나 상태 격자 없이, 학생의16단계 제어와 교사의 실제 남은 비용을 연결하는 알고리즘을 구현하고 평가했다. 두 문제군×5시드×5손실의50개 학습을 수행했다. 학습은32차원에서만 했고, 같은 신경망 가중치를64·128·256차원으로 옮겼다. 독립 초기상태는6144개다.','',
        '결과는 고전 탐색 대비 학습의 효과와, Bellman 학습이 다른 학습법보다 주는 추가 효과를 구분해서 읽어야 한다. 아래 고전 비교군은 모두 실제로 실행했으며, 이론상 격자 저장량만을 상대하지 않았다.','',
        '| 문제 | 차원 | GPU 고전 탐색 | Bellman 학생 | 직접 정책 최적화 | 교사 없는 짧은 학습 |','|---|---:|---:|---:|---:|---:|']
    for fam in ['mechanical','reaction']:
        for c in nominal(fam): ko.append('| '+('기계계' if fam=='mechanical' else '반응–확산')+f' | {c["dimension"]} | '+ ' | '.join(f'{c["means"][m]:.4f}' for m in ['best_classical','block16','dpc','truncated16'])+' |')
    ko+=['','비용은 작을수록 좋다. 신경망 수치는5개 시드의 평균이며, 각 행에서 동일한512개 초기상태를 사용했다. GPU 고전 탐색은4개 초기값의1024회 Adam 및 CEM+Adam 결과 중 사례별 최선이다. 전역 최적값이라는 뜻은 아니다.','',
        '## 더 강한 CPU 비교','',
        'CPU L-BFGS를 LQR·0입력·GPU 탐색 최선값에서 각각1024회까지 실행했다. 계산량이 큰 비교라 사전에 정한128개 nominal /64개 shifted 상태를 사용했다. 아래 개선율과 신뢰구간은 같은 상태끼리 비교하며, 음수이면 학생의 비용이 더 높다.','',
        '| 문제 | 차원 | CPU/GPU 대비 학생 비용 감소율 | 95% 구간 |','|---|---:|---:|---:|']
    for fam in ['mechanical','reaction']:
        for c in nominal(fam):
            v=c['native_subset']['hybrid_comparison']; ko.append(f'| {fam} | {c["dimension"]} | {v["reduction_percent"]:.2f}% | [{v["ci95_percent"][0]:.2f}, {v["ci95_percent"][1]:.2f}]% |')
    memory_savings={f:100*(1-s['isolated_memory'][f'{f}_block16']['incremental_peak_bytes']/s['isolated_memory'][f'{f}_dpc']['incremental_peak_bytes']) for f in ['mechanical','reaction']}
    nt=s['timing']['mechanical_d256']['methods']
    ko+=['','## 계산과 한계','',f'- 전체 준비·설정·학습·검증 시간은 DPC 대비 기계계{savings["mechanical"]:.1f}%, 반응–확산계{savings["reaction"]:.1f}% 줄었다. 각 방법의 실제 비용과 별도 프로세스 메모리를 원자료에 기록했다.',
        f'- 별도 프로세스에서 측정한 학습 그래프·신경망·최적화기 메모리는 각각{memory_savings["mechanical"]:.1f}%, {memory_savings["reaction"]:.1f}% 줄었다. 데이터와 CUDA 드라이버 메모리는 제외한 수치다.',
        f'- 기계계256차원에서 동일한40단계 계획 생성은 학생{nt["block16_full_plan"]["wall_median_ms"]:.2f}ms, 순차 CPU 두 초기값 최적화{nt["native_two_start1024"]["wall_median_ms"]:.1f}ms였다. 후자는 위 표의 더 강한3시작 CPU/GPU 혼합 탐색과 별도 비교군이다.',
        '- Bellman 방식은 신경망을16단계만 미분하지만, 남은 교사 시뮬레이션과 상태 미분 계산은 여전히 수행한다. 교사 계산을 무료로 취급하지 않았다.',
        '- 기계계의 조건 변경에서는 DPC와 교사 없는 짧은 학습보다 Bellman 학생이 나빠지는 경우가 있다. 기본 제어기 대비 개선을 모든 비교군 대비 개선으로 바꾸어 말할 수 없다.',
        f'- 사전에 정한 엄격한 HJ 고유 기여 기준은16개 조건 중{failures}개에서 충족했다. 더 작은 계산–품질 절충을 이 기준의 성공으로 재분류하지 않았다.',
        '- 신뢰구간은5개 학습 시드와 공통 테스트 상태를 따로 재표집했다. 시드×상태를 모두 독립 학습으로 세지 않았으며, 다중 비교 보정은 하지 않았다.',
        '- 모든 제어기는 같은 공유 입력 예산을 사용한다. float64 독립 비용 재생·기울기·Bellman 항등식·최적화기 초기화·제약 검사를 수행했다.',
        '- 이 결과는 균질한 결합 고리 두 종류의 구조를 이용한다. 일반 고차원 HJB의 차원의 저주 해결, 실제 설비 검증, 전역 최적성, 상태 안전성은 주장하지 않는다.','',
        '## 논문 메시지','',
        '**이론 메시지:** teacher–student 제어 학습은 HJB의 정책 평가와 개선을 연결한다. 교사가 미래 비용의 가치함수를 제공하면 학생은 그 가치함수를 기준으로 Hamiltonian 최소화 단계를 근사한다. 교사에게 남은 개선 여지는 교사 가치함수의 HJB residual의 음수이며, 총비용 차이는 학생 경로에서 누적된 선택 오차에서 이 개선 여지를 뺀 값이다. 개선된 정책의 재평가는 다음 정책 반복 단계에 해당한다. [HJB 연결 해설](hj_gridfree_hjb_connection_ko.md), [전체 이론 해설](hj_gridfree_framework_ko.md)에 식과 예제를 정리했다.','',
        '**실험 메시지:** 짧은 학생 구간과 교사 continuation으로 계산–품질 절충을 구현했고, 그 효과와 한계를 측정했다. 사전 성능 기준 미달을 이론의 설명력과 혼동하지 않되, 강화된 이론 서술을 실험 성공으로 재분류하지 않는다. Hamiltonian 기반 모방과 정책 개선은 선행연구에 존재한다. 현재 기여는 연결된 해석과 구현·검증이며, 일반적인 HJ 연결을 처음 발견했다거나 고전 알고리즘을 전반적으로 압도했다는 주장은 하지 않는다.','',
        '[영문 원고 PDF](../build/hj_gridfree_rollout.pdf), [고정 실험 계획](hj_gridfree_experiment_protocol.md), [이론](hj_gridfree_theory.md), [전체 통계](../experiments/results/hj_gridfree_primary_v1/report/summary.json).']
    Path('docs/hj_gridfree_results_ko.md').write_text('\n'.join(ko)+'\n',encoding='utf-8')
    print('Paper sections and Korean results written.',flush=True)


if __name__=='__main__': main()
