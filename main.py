import sys
import argbind
import warnings
import traceback
from src import train
from src import validate
from src import test

STAGES = ["train", "validate", "test"]
warnings.filterwarnings("ignore")

def run(stage: str):
    if stage not in STAGES:
        raise ValueError(f"Unknown command: {stage}. Allowed commands are {STAGES}")

    stage_fn = globals()[stage]
    stage_fn("config.yaml")

if __name__ == "__main__":
    try:
        group = sys.argv.pop(1)
        args = argbind.parse_args(group=group)
        with argbind.scope(args):
            run(group)
    except Exception as e:
        print('\n'+str(e))
        print(traceback.format_exc())