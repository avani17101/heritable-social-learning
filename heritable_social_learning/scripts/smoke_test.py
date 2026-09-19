import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT,cli

if __name__=='__main__':
    args=cli('Run smoke and invariant tests').parse_args()
    result=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'tests'),'-q'],capture_output=True,text=True)
    output=args.output or ROOT/'results'
    output.mkdir(parents=True,exist_ok=True)
    text=result.stdout+result.stderr
    (output/'smoke_test.txt').write_text(text)
    print(text)
    raise SystemExit(result.returncode)
