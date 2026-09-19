"""Re-evaluate saved champions; never rerun selection or choose by transfer score."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate
from src.utils import ROOT,cli,load_config,provenance,save_frame,setup

if __name__=='__main__':
    args=cli(__doc__).parse_args()
    setup()
    config=load_config(args.config)
    output=args.output or ROOT/'results'/Path(args.config).stem
    saved=json.loads((output/f'seed_{args.seed:03d}_{args.condition}'/'complete.json').read_text())
    if saved['config']!=config:
        raise ValueError('Transfer config differs from saved evolutionary run')
    dest=output/f'transfer_{args.seed:03d}_{args.condition}'
    dest.mkdir(exist_ok=True)
    if any(dest.iterdir()) and not args.fresh:
        raise FileExistsError('Transfer output exists; use --fresh')
    meta=provenance(config,output)
    meta['champion_source_hash']=saved['source_hash']
    for name,rows in zip(['episodes','summary','diagnostics','genomes'],evaluate(config,args.seed,saved['champions'])):
        save_frame(rows,dest/f'{name}.parquet',meta)
    print(dest)
