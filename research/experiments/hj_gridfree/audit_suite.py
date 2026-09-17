import subprocess
import sys


def main():
    for family in ['mechanical','reaction']:
        for mode in ['block16','dpc','truncated16','mse','bc']:
            subprocess.run([sys.executable,'-B','-m','experiments.hj_gridfree.memory','--family',family,'--mode',mode],check=True)
    subprocess.run([sys.executable,'-B','-m','experiments.hj_gridfree.batch_audit'],check=True)
    print('MEMORY AND BATCH AUDITS COMPLETE',flush=True)


if __name__=='__main__': main()
