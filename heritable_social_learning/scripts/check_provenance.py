import json
import sys
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,canonical_hash,cli


def check(output):
    manifests={}
    for path in (output/'provenance').glob('*.json'):
        m=json.loads(path.read_text())
        assert canonical_hash(m['files'])==m['source_hash'],path
        assert canonical_hash(m['config'])==m['config_hash'],path
        bundle=path.parent/f"{m['source_hash']}.bundle"
        if not m.get('git_dirty',True):
            assert bundle.exists(),path
            with tempfile.TemporaryDirectory(prefix='verify-social-source-') as temporary:
                subprocess.run(['git','clone','-q',str(bundle.resolve()),temporary],check=True,capture_output=True)
                actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=temporary,text=True).strip()
                assert actual==m['git_sha'],path
                for name,content in m['files'].items():
                    assert (Path(temporary)/name).read_text()==content,(path,name)
        manifests[(m['source_hash'],m['config_hash'])]=m
    assert manifests,'No source/config manifests'
    baseline=json.loads((output/'baseline.json').read_text())
    frozen=baseline.pop('freeze_hash')
    assert canonical_hash(baseline)==frozen
    checked=0
    for path in output.glob('**/*.parquet'):
        if path.parent.name=='analysis':
            continue
        frame=pd.read_parquet(path)
        assert not frame.empty,path
        for source,config in frame[['source_hash','config_hash']].drop_duplicates().itertuples(index=False,name=None):
            assert (source,config) in manifests,path
        if 'baseline_hash' in frame:
            assert (frame['baseline_hash']==frozen).all(),path
        keys=[k for k in ['seed','condition','generation','phase','regime','episode','agent','slot','split','candidate'] if k in frame]
        assert not frame.duplicated(keys).any(),(path,keys)
        checked+=len(frame)
    for path in output.glob('seed_*/complete.json'):
        run=json.loads(path.read_text())
        assert run['config_hash']==canonical_hash(run['config']),path
        assert run['baseline_hash']==frozen,path
    print(f'PASS: {checked:,} source rows; {len(manifests)} verified content-addressed manifests; frozen baseline intact')
    return checked


if __name__=='__main__':
    args=cli('Check immutable source snapshots and every numeric source row').parse_args()
    check(args.output or ROOT/'results'/Path(args.config).stem)
