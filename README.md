# Dissertation (LaTeX)

This repository contains the LaTeX source for my dissertation. The project is structured around KOMA-Script (`scrreport`) and a University of Konstanz corporate design theme (`utilities/themeKonstanz*`).

## Current status

`thesis.tex` is the main entry point and currently compiles theory, methods, and the results chapters (tunnel barrier and atomic contact).

- Theory, methods, and the tunnel-barrier / atomic-contact results chapters have all gone through a first round of feedback (Elke on theory and atomic contact, Ronja on tunnel barrier, Lukas on SSN-SET and cryo, Valentin on setup, Daniel on conclusion), which has been incorporated.
- Currently in a polishing/formatting pass (spacing, references, figures, TOC/appendix style) ahead of submission.
- The SSN-SET chapter (`ssn-set/`) is written and included, but currently sits in the post-body/appendix section rather than the main results flow.

Key dates: Abgabe (submission) 2026-08-24, Gutachten deadline 2026-09-03, mündliche Prüfung 2026-10-08.

## Repository structure

- `thesis.tex`
  Main file controlling the build and which chapters are included.

- `header.tex` (via `utilities/header.tex`)
  Central preamble: page geometry (A5, do not change; enlarge manually to A4 for Prüfungsamt submission), corporate design theme, math packages, plotting/graphics support, and `biblatex` configuration (`backend=biber`).

- `theory/`
  Theory chapters. Contains many Matplotlib-PGF figures (`.pgf`) and supporting graphics.
  Notable files:
  - `theory/theory.tex` (chapter header)
  - `theory/basics.tex`, `theory/micro.tex`, `theory/macro.tex`, `theory/meso.tex`, `theory/stochastic.tex`
  - `theory/diffusive.tex` (included in the post-body/appendix section)

- `methods/`
  Experimental methods chapter. Notable files:
  - `methods/methods.tex` (chapter header)
  - `methods/sample.tex`, `methods/setup.tex`, `methods/digital.tex`
  - `methods/fabrication.tex`, `methods/cryo.tex` (post-body/appendix)
  Also contains many figure assets (PNG/PDF/SVG/PGF, and some `pdf_tex` exports).

- `tunnelbarrier/`
  Tunnel-barrier results chapter (`tunnelbarrier.tex`, `highres.tex`, `asym.tex`) plus `appendix.tex`.

- `atomic_contact/`
  Atomic-contact results chapter (`atomic-contact.tex`, `pincode.tex`, `photon-assisted.tex`, `subharmonic.tex`) plus `appendix.tex`.

- `ssn-set/`
  Superconductor-superconductor-normal single-electron transistor (SSN-SET) chapter (`ssn-set.tex`, `static.tex`, `dynamic.tex`), currently included in the post-body section.

- `dcb-ssn/`
  Raw/working data folders for dynamical-Coulomb-blockade SSN measurements, not directly referenced from `thesis.tex`.

- `miscellaneous/`
  Front/back matter and shared source: titlepage, abstract, introduction, results intro, conclusion, references, software, acknowledgements, `eidestattliche-versicherung`, plus the bibliography files (`local.bib`, `intro.bib`).

- `utilities/`
  Style files (Konstanz theme), logos, and templates used by the document.

- `papierkram/`
  Administrative paperwork for the submission (signed declaration, front page, forms, correspondence) — not thesis source content.

- `versionen/`
  Dated PDF snapshots of chapters sent out for review; not build source.

- `output/`
  Rendered PDF output snapshots.

- `.auxiliary/`, `tmp/`
  Build artifacts and scratch files.

## Bibliography

The document uses `biblatex` with `biber`:

- Zotero bibliography file: `local.bib`
- Introduction-specific bibliography file: `intro.bib`
