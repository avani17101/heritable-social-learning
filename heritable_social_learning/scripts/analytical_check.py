import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment import analytical
from src.utils import cli

if __name__ == '__main__':
    args = cli('Exact analytical environment gate (no stochastic experimental parameters)').parse_args()
    result = [analytical(.7,.7), analytical(.85,.5), analytical(.5,.85)]
    assert result[0]['optimal'] > max(result[0]['private'],result[0]['blind_copy'])
    assert all(r['min_joint'] > 0 for r in result)
    assert result[1]['follow_peer_high'] < result[2]['follow_peer_high']
    folder = args.output or Path(__file__).resolve().parents[1] / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'analytical.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
