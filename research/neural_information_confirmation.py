"""Independent-seed confirmation of a secondary mechanism-study finding.

Conditions selected from the complete initial 240-run report. The initial
radius2/width2 primary is inconclusive; it is not replaced or pooled here.
This driver and the base implementation are hashed before any new training.
"""
import copy
from pathlib import Path
import neural_information_transfer as study

BASE=Path(study.__file__).resolve()
PARENT=study.OUT/'report.json'
study.OUT=study.HERE/'results/neural_information_confirmation'
study.CFG=copy.deepcopy(study.CFG)
study.CFG.update(seeds=list(range(12201,12211)),control_radii=[.5,2.,32.],
    primary='Independent confirmation: restored minus point held-out normalized local Bellman regret at radius0.5/width2, ten paired seeds, unadjusted paired95%t interval. Benefit requires upper endpoint below zero.',
    secondary='Width4 severe-constraint effect; radius2 boundary; width16 sufficient-capacity control; radius32 inactive-constraint equality; closed-loop cost; weight-only and shuffled-normal comparisons. All are secondary, with unadjusted intervals.',
    reporting='All360 independent confirmation runs retained, without pooling with the initial240 runs. Radius0.5 selected using the initial study secondary response. Initial radius2/width2 primary remains inconclusive. No new tuning; identical architecture, data law, optimizer and4096 updates.')

if __name__=='__main__':
    import json
    plan=dict(config=study.CFG,base_source_sha256=study.digest(BASE),
        driver_source_sha256=study.digest(Path(__file__)),parent_report_sha256=study.digest(PARENT),
        rationale='The initial study had a null primary at radius2/width2; severe radius0.5 improved regret in5/5 pairs at widths2 and4. Select that secondary condition prospectively for independent seeds, retain radius2 and capacity/inactive controls. No source or hyperparameter changes.')
    path=study.OUT/'confirmation_plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:study.write(path,plan)
    study.run()
