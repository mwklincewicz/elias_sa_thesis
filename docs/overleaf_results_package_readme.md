# Overleaf Results Outline Package

Upload this zip to Overleaf as a replacement project or use it to create a fresh Overleaf copy.

## What changed

- `main.tex` now inputs `results_outline.tex` at the Results section.
- `main.tex` now inputs `appendix_results_figures.tex` after the bibliography.
- `references.bib` includes the APA7-ready citation anchors for the Results chapter.
- `figures/results/main_text/` contains the ten Results figures.
- `figures/results/appendix/` contains supporting appendix figures.
- `results_section_outline_overleaf.md` is included as a fuller writing guide.
- `figure_manifest.csv` maps each packaged figure to its original source artifact.

## How to work in Overleaf

1. Open `results_outline.tex`.
2. Turn each bullet list into prose one subsection at a time.
3. Keep the figure environments in place unless a figure makes the chapter too long.
4. Move any extra figure to `appendix_results_figures.tex`.
5. Compile with Biber because the template uses `biblatex` APA style.

## Safety notes

- Do not claim WordPress accuracy, F1, or formal bias metrics unless gold labels are added.
- Do not claim SHAP, longitudinal modeling, weak-label training, or inter-annotator reliability unless those analyses are actually added.
- Keep Gemma Stories results framed as sentiment-only.
