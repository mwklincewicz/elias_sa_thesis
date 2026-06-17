PYTHON ?= python

.PHONY: help check compile test docs-check lemotif-eda bert-baseline bert-optuna stories wordpress-order wordpress-smoke

help:
	@echo "Code Repository Deliverable targets:"
	@echo "  make check            Run clean no-data repository checks"
	@echo "  make lemotif-eda      Rebuild Lemotif cleaning and EDA outputs"
	@echo "  make bert-baseline    Rebuild Lemotif cleaning, EDA, and BERT baseline"
	@echo "  make bert-optuna      Run Optuna BERT and diagnostic comparisons"
	@echo "  make stories          Run restricted Stories external-test workflow"
	@echo "  make wordpress-order  Print WordPress workflow order"
	@echo "  make wordpress-smoke  Run deterministic WordPress smoke test"

check: compile test docs-check

compile:
	$(PYTHON) -m compileall -q src thesis_findings_vscode/wordpress_findings_codebase

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

docs-check:
	$(PYTHON) scripts/check_crd_docs.py

lemotif-eda:
	$(PYTHON) src/run_pipeline.py

bert-baseline:
	LEMOTIF_RUN_MODEL=1 $(PYTHON) src/run_pipeline.py

bert-optuna:
	$(PYTHON) src/train_bert_optuna.py
	$(PYTHON) src/model_eda/run_optuna_comparison.py
	$(PYTHON) src/model_eda/run_bert_optuna_diagnostics.py

stories:
	$(PYTHON) src/test_data/run_all.py

wordpress-order:
	cd thesis_findings_vscode/wordpress_findings_codebase && $(PYTHON) run_wordpress_findings.py recommended-order

wordpress-smoke:
	cd thesis_findings_vscode/wordpress_findings_codebase && $(PYTHON) run_wordpress_findings.py smoke
