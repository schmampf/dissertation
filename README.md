# Dissertation (LaTeX)

This repository contains the LaTeX source for my dissertation. The project is structured around KOMA-Script (`scrreport`) and a University of Konstanz corporate design theme (`.utilities/themeKonstanz*`).

## Current status

`thesis.tex` is the main entry point. At the moment, it compiles only the theory chapter:

- `\include{theory/theory}` is enabled
- `methods/`, `results/`, and several `miscellaneous/` chapters exist but are currently commented out in `thesis.tex`

To compile additional parts, uncomment the relevant `\include{...}` lines in `thesis.tex`.

## Repository structure

- `thesis.tex`  
  Main file controlling the build and which chapters are included.

- `header.tex`  
  Central preamble: page geometry (A5), corporate design theme, math packages, plotting/graphics support, and `biblatex` configuration (`backend=biber`).

- `theory/`  
  Theory chapter(s). Contains many Matplotlib-PGF figures (`.pgf`) and supporting graphics.  
  Notable files:
  - `theory/theory.tex` (chapter driver; currently included from `thesis.tex`)
  - `theory/basics.tex` (basic concepts / normal-state + mesoscopic preliminaries)
  - `theory/micro.tex` (microscopic superconductivity / BCS-level building blocks)
  - `theory/macro.tex` (macroscopic superconductivity / phase, Josephson relations, electrodynamics)
  - `theory/meso.tex` (mesoscopic superconducting transport perspective)
  - `theory/stochastic.tex` (stochastic/finite-temperature aspects; noise / fluctuations where applicable)
  Additional / work-in-progress material:
  - `theory/mesowave.tex` (microwave-driven transport notes / standalone draft)
  - `theory/todo.tex` (scratchpad / placeholders; included currently, so expect unfinished parts)
  
- `methods/`  
  Experimental methods chapter scaffold. Notable files:
  - `methods/methods.tex` (chapter header)
  - `methods/sample.tex` (sample preparation section)
  - `methods/appendix.tex` (step-by-step fabrication appendix-style content)
  Also contains many figure assets (PNG/PDF/SVG/PGF, and some `pdf_tex` exports).

- `results/`  
  Results chapter (currently not included in `thesis.tex`).

- `miscellaneous/`  
  Optional chapters/sections (e.g., abstract, intro, conclusion, appendix, acknowledgements), depending on what is included from `thesis.tex`.

- `.utilities/`  
  Style files (Konstanz theme), logos, and templates used by the document.

## Bibliography

The document uses `biblatex` with `biber`:

- Default bibliography file: `My Library.bib`