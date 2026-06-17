# Data Folder

This public repository does not include raw thesis datasets. The project is split this way so the code and methodology can be reviewed publicly while restricted or sensitive reflection data remains outside Git.

Expected local files for full reproduction:

- `Lemotif thesis data.csv`: Lemotif training dataset. Acquire through the approved Lemotif source/access channel described in the thesis.
- `stories_source.xlsx`: restricted Stories/University of Twente evaluation workbook. This file is private and must only be placed here by authorized reviewers.

After placing approved local files here, run:

```powershell
python src/clean_data.py
python src/test_data/prepare_stories_data.py
```

For the finalized Methodology Overview and the full public/restricted reproducibility boundary, see [`../docs/METHODOLOGY_OVERVIEW.md`](../docs/METHODOLOGY_OVERVIEW.md).
