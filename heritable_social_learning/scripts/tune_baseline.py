import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment import tune
from src.utils import ROOT,cli,load_config

if __name__=='__main__':
    args=cli('Tune on training/validation, never test environments').parse_args()
    config=load_config(args.config)
    print(json.dumps(tune(config,args.output or ROOT/'results'/Path(args.config).stem,args.fresh),indent=2))
