import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiment import run
from src.utils import ROOT,cli,load_config

if __name__=='__main__':
    parser=cli('Run independent evolutionary seeds and fresh newborn evaluations')
    parser.add_argument('--all-seeds',action='store_true')
    args=parser.parse_args()
    config=load_config(args.config)
    output=args.output or ROOT/'results'/Path(args.config).stem
    for seed in range(config['seeds']) if args.all_seeds else [args.seed]:
        result=run(config,seed,args.condition,output,args.fresh)
        print(f"seed={seed} condition={args.condition} completed in {result['elapsed_seconds']:.1f}s",flush=True)
