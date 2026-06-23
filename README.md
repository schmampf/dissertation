# Dissertation (LaTeX)

This repository contains the LaTeX source for my dissertation. The project is structured around KOMA-Script (`scrreport`) and a University of Konstanz corporate design theme (`utilities/themeKonstanz*`).

## Current status

`thesis.tex` is the main entry point. At the moment, it compiles only the theory chapter:

- Theory close to be finished / ready for first pass to Elke.
- Methods sent to Elke. Waiting for feedback.


## Repository structure

- `thesis.tex`  
  Main file controlling the build and which chapters are included.

- `header.tex`  
  Central preamble: page geometry (A5), everything is now in accordance with A5. Dont change. For Prüfungsamt, enlarge manually to a4. Further includes: corporate design theme, math packages, plotting/graphics support, and `biblatex` configuration (`backend=biber`).

- `theory/`  
  Theory chapter(s). Contains many Matplotlib-PGF figures (`.pgf`) and supporting graphics.  
  Notable files:
  - `theory/theory.tex` (chapter header)
  - `theory/basics.tex` (basic concepts / normal-state + mesoscopic preliminaries)
  - `theory/micro.tex` (microscopic superconductivity / BCS-level building blocks)
  - `theory/macro.tex` (macroscopic superconductivity / phase, Josephson relations, electrodynamics)
  - `theory/meso.tex` (mesoscopic superconducting transport perspective)
  - `theory/stochastic.tex` (stochastic/finite-temperature aspects; noise / fluctuations where applicable)
  
- `methods/`  
  Experimental methods chapter scaffold. Notable files:
  - `methods/methods.tex` (chapter header)
  - `methods/sample.tex` (sample preparation section)
  - `methods/setup.tex` (physical setup)
  - `methods/digital.tex` (digital steps involved in data acquisition and evaluation)
  - `methods/sampleappendix.tex` (step-by-step fabrication appendix-style content)
  Also contains many figure assets (PNG/PDF/SVG/PGF, and some `pdf_tex` exports).

- `results/`  
  Results chapter (currently not included in `thesis.tex`).

- `miscellaneous/`  
  Optional chapters/sections (e.g., abstract, intro, conclusion, appendix, acknowledgements), depending on what is included from `thesis.tex`.

- `utilities/`  
  Style files (Konstanz theme), logos, and templates used by the document.

## Bibliography

The document uses `biblatex` with `biber`:

- Zotero bibliography file: `My Library.bib`
- Local bibliography file: `local.bib`
