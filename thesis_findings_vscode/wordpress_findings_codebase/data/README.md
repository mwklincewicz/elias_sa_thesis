# WordPress Selected-Site Data

This public code package includes `selected_sites.txt`, the selected public WordPress site list used by the sanity-check pipeline.

Full scraped post text is not committed. Regenerate it locally with:

```powershell
python run_wordpress_findings.py selected-sites
```

The selected-site corpus is unannotated and should be treated as descriptive sanity-check evidence, not a formal accuracy dataset.
