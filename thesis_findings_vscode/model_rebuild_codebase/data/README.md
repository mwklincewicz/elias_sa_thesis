# Data Folder

This public code package does not include raw thesis datasets or private evaluation data.

For full model rebuilds, place approved local copies here:

- `Lemotif thesis data.csv`
- `stories_source.xlsx`

The rebuild runner will then prepare cleaned/generated files with:

```powershell
python run_models.py prepare-data
```
