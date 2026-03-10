# Writing Profile and Tone Guide

This file captures the preferred writing style for this dissertation project.
Use it as a reference for drafting, rewriting, and language correction.

## Scope

- Applies to all chapters in this repository.
- Note: `theory/stochastic.tex` is currently an early draft and should not define final style.

## Core Preferences

- Build a clear regime-based story (tunnel, finite transmission, single-electron/stochastic).
- Always connect theory to observables (`\textit{I--V}`, `\textit{dI--dV}`, maps, extracted parameters).
- State assumptions and model validity limits explicitly.
- Keep engineering quality and reproducibility visible (setup changes, processing pipeline, uncertainties).

## Language, Voice, and Tense

- Use US English.
- Use "we" in theory chapters.
- Use "I" in methods, results, and other non-theory chapters.
- Use present tense for theory statements.
- Use past tense for performed experiments and completed analysis steps.

## Preferred Chapter Pattern

1. Physical question and measurement context.
2. Data presentation (clean, representative figures).
3. Data treatment and extraction method.
4. Minimal model comparison.
5. What is explained well.
6. What remains open or deferred.

## Language Style

- Concept-first, technical, and structured.
- Use short signposting transitions ("In this section...", "We now compare...", "This implies...").
- Keep notation consistent across chapters.
- Prefer concise sentences over long nested constructions.
- Remove repetition unless it improves clarity.
- Avoid double-dash or em-dash style punctuation.
- Minimize parentheses. Prefer short full-stop sentences.
- Avoid semicolons. Prefer commas and short full-stop sentences.
- Use footnotes for short side explanations and rough order-of-magnitude estimates of physical quantities.

## Terminology Consistency (target)

- Use one form consistently for each term (examples):
  - quasiparticle (not mixed with quasi-particle)
  - Cooper pair as noun. Cooper-pair only as compound adjective before a noun.
  - normal-state
  - weak-coupling
  - measurement-name style: `\textit{I--V}` and `\textit{dI--dV}`
  - function style in equations: `$I(V)$` and `$\mathrm{d}I/\mathrm{d}V$`

## Citation Placement

- For a single sentence-level claim from a source, place citation at sentence end before the period.
- For a paragraph mainly based on one or more sources, place citation at paragraph end after the period.
- For an introductory paragraph where specific papers frame the whole chapter, place citation at the end of that introductory paragraph.
- For equations, numeric values, or non-obvious approximations taken from literature, cite immediately at that local claim.
- For mixed claims from different papers in one paragraph, cite each claim locally instead of only one citation at paragraph end.

## Practical Editing Rules

- Every subsection should end with a one-paragraph takeaway.
- Every model section should include a short "scope/limits" paragraph.
- If a section is incomplete, keep explicit TODO bullets instead of vague filler text.
- Move lab-manual detail to appendices; keep main text focused on reproducible method and interpretation.

## Figure Caption Template

- Sentence 1. State the main physical result shown in the figure.
- Sentence 2. State the main trend or mechanism.
- Sentence 3. Give only essential conditions or parameters.
- Keep panel-specific details in subcaptions whenever possible.
- Avoid repeating full derivations from the main text.
