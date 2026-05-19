# Thesis Findings VS Code Workspace

This folder contains four small, runnable thesis codebases:

- `primary_findings_codebase`: model performance, Stories external test, and primary RQ1 evidence.
- `secondary_findings_codebase`: dataset EDA, grouped emotion analysis, dataset comparison, and optional supporting evidence.
- `model_rebuild_codebase`: one-by-one model rebuild commands for reproducibility review.
- `wordpress_findings_codebase`: public WordPress reflection scraper, selected-site sanity-check analysis, and aggregate WordPress figures.

Open `thesis_findings.code-workspace` in VS Code to see both projects side by side.

## Quick Run

Primary findings, using bundled lightweight result artifacts:

```powershell
cd primary_findings_codebase
python run_primary_findings.py
```

Primary findings, rebuilding BERT artifacts:

```powershell
cd primary_findings_codebase
python run_primary_findings.py --full
```

Secondary findings:

```powershell
cd secondary_findings_codebase
python run_secondary_findings.py
```

List model rebuild tasks:

```powershell
cd model_rebuild_codebase
python run_models.py list
```

Run one model rebuild step:

```powershell
cd model_rebuild_codebase
python run_models.py bert-baseline
```

WordPress public-reflection smoke test:

```powershell
cd wordpress_findings_codebase
python run_wordpress_findings.py smoke
```

Each codebase has its own `requirements.txt`, README, `.vscode/tasks.json`, and output folders. Raw/private data, generated model checkpoints, and generated output folders are intentionally excluded from the public repository. The WordPress codebase includes the selected-site list and regeneration commands, but not full scraped public-text tables.
