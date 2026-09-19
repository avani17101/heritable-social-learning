PYTHON ?= ../.venv/bin/python
CONFIG ?= configs/final.yaml

.PHONY: smoke debug tune final analyze check-provenance
smoke:
	$(PYTHON) scripts/analytical_check.py
	$(PYTHON) scripts/smoke_test.py
debug:
	$(PYTHON) scripts/tune_baseline.py --config configs/debug.yaml
	$(PYTHON) scripts/run_experiment.py --config configs/debug.yaml --all-seeds
	$(PYTHON) scripts/analyze_results.py --config configs/debug.yaml
tune:
	$(PYTHON) scripts/tune_baseline.py --config $(CONFIG)
final:
	$(PYTHON) scripts/run_experiment.py --config $(CONFIG) --all-seeds
analyze:
	$(PYTHON) scripts/analyze_results.py --config $(CONFIG)
check-provenance:
	$(PYTHON) scripts/check_provenance.py --config $(CONFIG)
