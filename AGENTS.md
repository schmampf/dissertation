# Thesis Maintenance Guide

This repository is a modular LaTeX dissertation project. Use these rules when editing, restructuring, or extending the thesis. This file is the primary agent guide for both repository structure and writing style.

## Scope

- Apply these rules to all chapters in this repository.
- Treat `thesis.tex` as the only compile root.
- Treat `header.tex` as the only shared preamble.
- Keep chapter content in chapter-local `.tex` files and chapter-local asset folders.
- Do not treat `theory/stochastic.tex` as the style baseline for the rest of the thesis. It is still an early draft.

## Document Layout

- `thesis.tex` owns the document class, bibliography registration, page-numbering transitions, and top-level `\include{...}` switches.
- Chapter driver files such as `theory/theory.tex`, `methods/methods.tex`, and `results/results.tex` own one `\chapter{...}` and then pull in internal section files with `\input{...}`.
- Do not place a second preamble, `\documentclass`, or bibliography setup in chapter files.
- Keep optional chapters commented in `thesis.tex` until they are ready to compile cleanly.

## Preamble Rules

- Add packages, macros, geometry, float definitions, and shared typography settings in `header.tex`.
- Do not add ad hoc package imports inside chapter files unless there is a strong local reason and it does not duplicate shared setup.
- Preserve the Konstanz theme stack in `utilities/` unless the task is explicitly a theme change.

## File and Asset Placement

- Keep figures and raw assets close to the text that uses them.
- Use chapter-local directories like `theory/micro/`, `theory/schema/...`, `results/tunnelbarrier/...`, or `methods/sample/...`.
- Prefer `\import{...}{...}` for PGF or `pdf_tex` assets that live outside the current file directory.
- Use `\includegraphics{...}` for ordinary raster or PDF assets when no `\import` behavior is needed.
- Do not create a global catch-all figure folder unless there is a specific reorganization request.

## Structural Style

- Keep the visual section separators:

```tex
%=========================================================
```

- Use them around chapters, sections, and major `\input` boundaries.
- Preserve the existing indentation style inside environments and section bodies.
- Keep one responsibility per file: chapter driver, section text, or asset support file.

## Labels and Cross-References

- Add a `\label{...}` immediately after each chapter, section, subsection, figure, and important equation.
- Use namespaced labels:
  - `ch:...` for chapters
  - `sec:...` for sections
  - `subsec:...` for subsections
  - `fig:...` for figures
  - `eq:...` for equations
- Include chapter/topic context in label tails, for example `subsec:results:tb:model` or `eq:micro:tunnel`.
- Use `Sec.~\ref{...}`, `Chapter~\ref{...}`, `Fig.~\ref{...}`, and `Eq.~\eqref{...}` consistently.
  - When the complete equation reference is already enclosed in parentheses, use `(Eq.~\ref{...})` or `(Eqs.~\ref{...} and \ref{...})` to avoid nested parentheses such as `(Eq. (45))`.
  - Except its in the beginning of a sentence, then dont use the Abbreviation.

## Writing Conventions

- Use US English.
- Use present tense and `we` in theory chapters.
- Use past tense and `I` in methods, results, and other experiment-facing chapters.
- Keep the theory organized by regime and model level: basics, microscopic, macroscopic, mesoscopic, stochastic.
- Build a clear regime-based story across the thesis: tunnel, finite transmission, single-electron, and stochastic transport.
- Keep engineering quality and reproducibility visible, especially setup changes, processing steps, calibrations, and uncertainties.
- Always connect theory to observables such as `\textit{I--V}`, `\textit{dI--dV}`, maps, and extracted parameters.
- State scope, assumptions, and model limits explicitly.

## Chapter Pattern

- Prefer this chapter and section flow when writing results-like material:
  1. Physical question and measurement context.
  2. Data presentation with clean, representative figures.
  3. Data treatment and extraction method.
  4. Minimal model comparison.
  5. What is explained well.
  6. What remains open or deferred.
- For results chapters, keep this structure visible even when a section is still incomplete.
- For theory chapters, keep the same logic where possible: define the question, introduce the minimal model, connect to observables, and state limits.

## Language Style

- Write concept-first, technical, and structured prose.
- Keep each prose paragraph on one physical source line. Do not insert hard line breaks within a paragraph. Use a blank line only to separate paragraphs.
- Use short signposting transitions such as `In this section...`, `We now compare...`, and `This implies...` when they improve flow.
- Prefer concise sentences over long nested constructions.
- Remove repetition unless it materially improves clarity.
- Avoid double-dash or em-dash punctuation.
- Minimize parentheses. Prefer short full-stop sentences.
- Avoid semicolons. Prefer commas and short full-stop sentences.
- Use footnotes for short side explanations and rough order-of-magnitude estimates of physical quantities.
- Follow the thesis minimal-hyphen house style. Do not hyphenate a compound modifier whose first element is a noun, for example `microwave driven transport`, `photon assisted tunneling`, `sample dependent variation`, `Bessel weighted channels`, and `voltage biased response`.
- Hyphenate adjective-first compound modifiers before a noun when needed for clarity, for example `high-frequency response`, `low-temperature measurement`, and `normal-state conductance`. Omit the hyphen when the expression is used predicatively, for example `the response is frequency dependent`.
- Leave ordinary noun compounds unhyphenated, for example `microwave coupling`, `amplitude calibration`, `contact configuration`, and `photon energy`. Preserve hyphens in established names, numerical compounds, and constructions where omission would create genuine ambiguity. Prefer rephrasing over adding a noun-first adjectival hyphen.
- Use open noun compounds for named theories and models, with title capitalization for the complete name, for example `\(P(E)\) Theory` and `Tien-Gordon Model`. Do not join the name and the words `Theory` or `Model` with a hyphen or en dash.
- Use a single hyphen inside compound names and joint attributions, for example `Tien-Gordon Model` and `Josephson-quasiparticle cycle`. Reserve the LaTeX en dash `--` for relational constructions such as `\textit{I--V}`, `\textit{dI--dV}`, and `gate--bias map`.
- Write numerical ranges in math mode with a short text hyphen surrounded by thin spaces, for example `$1.8\,\text{-}\,5.0\,\mathrm{nm}$`. For named alphanumeric series in running text, use the corresponding text form, for example `M1\,-\,M4` and `C0\,-\,C3`. Do not use `-` or `--` as the range separator. Retain the LaTeX en dash for relational scientific notation such as `\textit{I--V}` and `gate--bias`.
- In running prose, write author abbreviations as `\textit{et al.}`. When another word follows directly, use a nonbreaking interword space, for example `Steinberg \textit{et al.}~reported`. When punctuation follows, attach it directly to the abbreviation, for example `Cuevas \textit{et al.}, which`.

## Terminology and Notation

- Prefer the forms below and keep them consistent:
  - `quasiparticle`
  - `Cooper pair` in both noun and compound modifier uses, for example `Cooper pair transfer`
  - `normal-state`
  - `weak-coupling`
  - `\textit{I--V}` and `\textit{dI--dV}` for measurement names
  - `\textit{P(E)} Theory`
  - `Tien-Gordon Model`
  - `I(V)` and `\mathrm{d}I/\mathrm{d}V` in equations
- Keep notation stable across chapters unless a local redefinition is unavoidable and clearly introduced.

## Citation Placement

- For a single sentence-level claim from a source, place the citation at sentence end before the period.
- For a paragraph mainly based on one or more sources, place the citation at paragraph end after the period.
- For an introductory paragraph where specific papers frame the whole section or chapter, place the citation at the end of that introductory paragraph.
- For equations, numeric values, or non-obvious approximations taken from the literature, cite immediately at the local claim.
- For mixed claims from different papers in one paragraph, cite each claim locally instead of relying on one paragraph-end citation.

## Figures and Captions

- Prefer representative, data-carrying figures over decorative figures.
- Keep panel-specific details in subcaptions when possible.
- Write captions with this structure:
  1. Main physical result.
  2. Main trend or mechanism.
  3. Essential conditions or parameters only.
- Avoid placeholder captions in finished sections.

## Incomplete Sections and TODOs

- Do not hide unfinished content behind vague prose.
- Use explicit source TODOs in the form `TODO[ID]`.
- Keep TODO IDs aligned with `todo.md` when the task is substantial or tracked across files.
- If a section is still a scaffold, leave a concise placeholder plus actionable TODO bullets.

## Practical Editing Rules

- Every subsection should end with a short takeaway paragraph unless there is a clear reason not to.
- Every model section should include a short scope or limits paragraph.
- Move lab-manual detail to appendices. Keep the main text focused on reproducible method and interpretation.
- When replacing placeholders, prefer real content over polished filler.

## Build and Generated Files

- Source files are the `.tex`, `.bib`, `.sty`, PGF, vector, and image assets.
- Command-line builds should use the project `latexmk` configuration so auxiliary files go to `.auxiliary/` while the final PDF stays at the repository root.
- Prefer `latexmk thesis.tex` from the repository root over ad hoc `xelatex` runs when building manually.
- Treat `thesis.aux`, `thesis.bcf`, `thesis.log`, `thesis.out`, `thesis.run.xml`, `thesis.toc`, and similar outputs as generated artifacts, not content to hand-edit.
- `thesis.pdf` is generated and already ignored.
- Do not commit or rely on temporary compile products as source of truth.

## Editing Priorities

- Prefer minimal, local edits that preserve the existing structure.
- Update the nearest relevant file instead of duplicating text elsewhere.
- When adding new content, follow the existing chapter hierarchy before inventing a new layout.
- Only modify `utilities/` when the task is explicitly about styling or shared macros.

## Current Caveats

- `README.md` may lag behind the active includes in `thesis.tex`. Use `thesis.tex` as the source of truth for what currently compiles.
- Some files are intentionally scaffolded. Preserve their structure, but replace placeholders with real content as sections mature.
