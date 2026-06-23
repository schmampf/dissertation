# Theory Chapter Master Plan

This file tracks the work required to bring Chapter 1, *Superconducting Transport*, from its current strong draft state to a final dissertation chapter. The current baseline is approximately 18,500 words over 63 rendered pages. The chapter compiles without undefined references or citations and was rated 8.4/10 on 2026-06-22.

## Definition of done

The theory chapter is complete when:

- [ ] The chapter presents one coherent regime-based argument from normal-state transport to tunnel, finite-transmission, single-electron, and stochastic transport.
- [ ] Every section states its physical question, assumptions, observable predictions, and model limits.
- [ ] Every subsection ends with a concise takeaway or a clear transition.
- [ ] Notation and terminology are unique and consistent across all theory files.
- [ ] Every important equation, figure, section, and subsection has a namespaced label and is referenced consistently.
- [ ] Every non-obvious equation, numerical value, and literature-derived approximation has an appropriate citation.
- [ ] Figures and captions communicate the physical result without requiring the main text.
- [ ] The chapter contains no placeholders, unresolved TODOs, avoidable repetition, or known technical ambiguity.
- [ ] `latexmk thesis.tex` completes without undefined references, undefined citations, missing glyphs, or chapter-content layout defects.

## Priority 1: Close the chapter-level argument

- [ ] [THM-01] Add chapter-level model orientation and experimental mapping.
  - Introduce the model hierarchy near the beginning of the chapter instead of adding a separate synthesis section at the end.
  - Use the orientation table to map measured features to microscopic, macroscopic, mesoscopic, and stochastic descriptions.
  - Explain that mesoscopic theory extends both the microscopic spectral description and the macroscopic phase description to finite transmission.
  - Explain that stochastic theory is selected by loss of phase memory and can use microscopic or mesoscopic rates as inputs.
  - Add only a short closing transition after the stochastic section if it is needed for flow.
  - Acceptance criterion: a reader can determine which model is used for a measured feature before looking up the detailed theory.

- [ ] [THM-02] Add a compact regime-selection table or decision diagram.
  - Include tunnel quasiparticles, Josephson phase dynamics, finite-transmission Andreev transport, environmental energy exchange, and charge-state transport.
  - Include the controlling scales or parameters, such as transmission, coherence length, $E_\mathrm{J}$, $E_\mathrm{C}$, impedance, temperature, and drive frequency.
  - Acceptance criterion: the table summarizes the chapter without introducing notation that is absent from the text.

- [ ] [THM-03] Tighten the chapter introduction.
  - State the chapter's physical question before the historical background.
  - Distinguish the normal-state foundation from the four superconducting descriptions.
  - State explicitly that the descriptions form a hierarchy of useful models rather than four independent theories.
  - Check that the advertised scope matches the material actually derived.

- [ ] [THM-04] Strengthen transitions between the five main sections.
  - Normal state to microscopic: from transmission and material parameters to superconducting spectral scales.
  - Microscopic to macroscopic: from order-parameter magnitude to condensate phase.
  - Macroscopic to mesoscopic: from a lumped tunnel junction to channel-resolved finite transmission.
  - Mesoscopic to stochastic: from coherent transport to environmental fluctuations and incoherent rates.

## Priority 2: Resolve conceptual and notation issues

- [ ] [THM-05] Eliminate the collision in the charging-energy and cutoff-energy notation.
  - Reserve $E_\mathrm{C}$ for charging energy throughout the chapter.
  - Rename the environmental cutoff energy currently written as $E_\mathrm{c}=\hbar\omega_\mathrm{c}$, for example to $E_\mathrm{cut}$ or $\hbar\omega_\mathrm{c}$.
  - Check all equations, captions, and cross-references after the change.

- [ ] [THM-06] Standardize the term for the transmission set $\{\tau_i\}$.
  - Choose either `mesoscopic PIN code` or `mesoscopic pincode` based on the preferred literature usage.
  - Apply the choice consistently in `quasi.tex`, `meso.tex`, captions, and later experimental chapters.

- [ ] [THM-07] Audit the meaning and casing of all energy scales.
  - Check $E_\mathrm{J}$, $E_\mathrm{C}$, $E_\mathrm{T}$, $\Delta$, cutoff energies, and photon energies.
  - Ensure that the same symbol never denotes two physical quantities.
  - Ensure that capitalization and roman subscripts are consistent.

- [ ] [THM-08] Audit charge-dependent notation across driven and stochastic transport.
  - Check $q=e$, $q=2e$, and $q=me$ conventions.
  - Check $R_q$, phase factors, Bessel arguments, sideband shifts, and sign conventions.
  - Verify that Tien--Gordon, Shapiro, PAMAR, and $P(E)$ expressions use compatible voltage and frequency conventions.

- [x] [THM-09] Verify the boundary between coherent and incoherent descriptions.
  - State when Josephson transport is treated coherently with phase dynamics and when Cooper-pair transfer is treated as an incoherent rate.
  - State when MAR/FCS rates may be inserted into a charge-state master equation.
  - Identify where dephasing, lifetime broadening, or environmental coupling invalidates a population-only treatment.

## Priority 3: Calibrate the technical depth

- [ ] [THM-10] Decide whether `Microscopic Description` is the correct section title.
  - Current content uses weak-coupling BCS results but does not derive the BCS Hamiltonian or self-consistent gap equation.
  - Either add the minimal derivation needed to justify the title or rename/frame the section as the microscopic spectral and tunnel-limit description.
  - Acceptance criterion: the section introduction accurately describes its derivational depth.

- [ ] [THM-11] Review the normal-state section for necessary versus textbook background.
  - Retain only material that supports later diffusion, Landauer transport, atomic contacts, or parameter extraction.
  - Compress derivations that are not used later.
  - Preserve the distinction between bulk conductivity and coherent channel transport.

- [ ] [THM-12] Review the macroscopic section for model consistency.
  - Separate ideal Josephson relations, the spectroscopy-oriented Shapiro ansatz, deterministic RCSJ dynamics, and phenomenological switching curves.
  - Make clear which plotted quantities are calculated and which are prescribed.
  - Check whether the nonlinear quasiparticle shunt is developed sufficiently for later use.

- [ ] [THM-13] Review the mesoscopic section as the chapter's central model chain.
  - Verify the progression BdG to BTK to ABS to MAR to FCS to PAMAR.
  - Remove any repeated introductions of transmission channels or effective charge.
  - Ensure that the practical PAMAR approximation is clearly separated from full double-Floquet theory.

- [ ] [THM-14] Perform a dedicated stochastic-section physics pass.
  - Verify the $P(E)$ normalization, detailed-balance convention, impedance definition, and resistance-quantum convention.
  - Check the Ohmic and resonant kernels and their stated limits.
  - Check DCB and incoherent Cooper-pair tunneling prefactors and sign conventions against the cited references.
  - Check that the SSET master-equation treatment is sufficient for the results chapter and does not become a detached review.

## Priority 4: Improve focus and readability

- [ ] [THM-15] Reduce chapter length without removing required theory.
  - Target repeated definitions, repeated scope statements, and textbook explanations that do not support later observables.
  - Prefer one authoritative explanation followed by cross-references.
  - Initial target: remove 10--15% of prose while preserving the model chain and all experimental links.

- [ ] [THM-16] Perform a paragraph-level argument pass.
  - Give each paragraph one clear purpose.
  - Put the physical claim before mathematical detail where possible.
  - Split long nested sentences.
  - Remove avoidable parentheses, semicolons, and dash punctuation.

- [ ] [THM-17] Check subsection endings.
  - Every subsection should end with the observable consequence, extracted parameter, model limit, or transition to the next model.
  - Avoid endings that merely restate the last equation.

- [ ] [THM-18] Standardize cross-reference prose.
  - Use `Sec.~\ref{...}`, `Chapter~\ref{...}`, `Fig.~\ref{...}`, and `Eq.~\eqref{...}`.
  - Spell out Section, Chapter, Figure, or Equation at the beginning of a sentence.
  - Avoid nested equation-reference parentheses.

## Priority 5: Figures and captions

- [ ] [THM-19] Audit every figure against the caption structure.
  - First sentence: main physical result.
  - Second sentence: mechanism or trend.
  - Final sentence: essential conditions and parameters only.
  - Move panel-specific details into subcaptions where practical.

- [ ] [THM-20] Verify that all plotted assumptions are stated.
  - Check material parameters, temperature, Dynes broadening, drive frequency, transmission values, damping, impedance, and normalization.
  - Ensure that global defaults in the chapter introduction are not contradicted silently.

- [ ] [THM-21] Distinguish calculated, schematic, and phenomenological figures.
  - Label energy-space schematics explicitly.
  - Identify phenomenological switching and driven curves as such.
  - Do not imply that prescribed switching currents are outputs of the deterministic RCSJ equation.

- [ ] [THM-22] Fix known caption and typography defects.
  - Correct `The parameters are $q=e$,and ...` in the photon-assisted tunneling drive figure.
  - Check gray/grey usage against US English.
  - Check panel references, punctuation, spacing, and units.

## Priority 6: Citations and factual verification

- [ ] [THM-23] Audit citations section by section.
  - Cite equations, numerical material parameters, non-obvious approximations, and literature-specific model claims locally.
  - Move paragraph-level citations after the period where one source supports the full paragraph.
  - Avoid citations that appear detached from the supported claim.

- [ ] [THM-24] Prefer primary sources for model-defining claims.
  - Use original papers for BCS, Josephson relations, Shapiro steps, BTK, MAR, FCS, Tien--Gordon, and $P(E)$ theory.
  - Use textbooks and reviews for synthesis and standard derivations.

- [ ] [THM-25] Verify all aluminum-specific numbers and experimental setup claims.
  - Check $\Delta_0$, $T_\mathrm{c}$, coherence lengths, mean free paths, pair-breaking frequency, and transmitted microwave bandwidth.
  - Distinguish representative bulk values from measured device-specific values.

## Priority 7: Final consistency and build pass

- [ ] [THM-26] Run the terminology checklist across all theory files.
  - `quasiparticle`
  - `Cooper pair` as a noun and `Cooper-pair` as an adjective
  - `normal-state`
  - `weak-coupling`
  - `\textit{I--V}` and `\textit{dI--dV}`
  - `I(V)` and `\mathrm{d}I/\mathrm{d}V` in equations

- [ ] [THM-27] Run a label and cross-reference audit.
  - Confirm labels immediately follow chapters, sections, subsections, figures, and important equations.
  - Confirm namespace consistency.
  - Remove unused labels and repair ambiguous label tails.

- [ ] [THM-28] Run a source-format audit.
  - Keep each prose paragraph on one physical source line.
  - Preserve section separators and indentation.
  - Remove trailing whitespace and accidental formatting inconsistencies.

- [ ] [THM-29] Resolve build warnings relevant to the final document.
  - Fix missing bibliography glyphs.
  - Inspect meaningful overfull and underfull boxes.
  - Confirm that no figure, caption, equation, header, or footer is clipped or poorly placed.

- [ ] [THM-30] Perform the final render review.
  - Build with `latexmk thesis.tex` from the repository root.
  - Render the complete theory chapter to images.
  - Inspect section openings, float placement, page balance, equations, captions, headers, and transitions.
  - Acceptance criterion: no known content, reference, typography, or layout defect remains.

## Currently commented-out theory material

All five theory inputs are currently enabled in `theory/theory.tex`. No complete theory section is disabled. The following commented fragments require an explicit decision before the final source cleanup:

- [ ] [THM-31] Resolve the disabled section-counter override in `theory/quasi.tex`.
  - The source contains `% \setcounter{section}{-1}` before the normal-state section.
  - Keep the present numbering unless the normal-state material is intentionally meant to appear as Section 1.0.
  - Remove the stale commented command after the numbering decision is final.

- [ ] [THM-32] Resolve the commented `General Tunnel Junction` heading in `theory/micro.tex`.
  - Decide whether the general tunnel-current derivation needs a visible unnumbered subsubsection heading.
  - Restore the heading only if it improves navigation and remains structurally consistent with the NIN, NIS, and SIS discussion.
  - Otherwise remove the commented heading and separator lines permanently.

- [ ] [THM-33] Resolve commented explanatory text in `theory/quasi.tex`.
  - Review the sentence connecting the normal-state Hamiltonian to the diagonal BdG operator.
  - Integrate the point into active prose if the later BdG transition is not already sufficiently explicit. Otherwise delete the commented sentence.
  - Review the commented figure-legend description and retain it only as source documentation if it remains useful for maintaining the scattering-regime schematic.

- [ ] [THM-34] Remove obsolete commented layout overrides after visual QA.
  - Review the disabled `\vspace` adjustment in `theory/meso.tex` and the disabled `\captionsetup` line in `theory/stochastic.tex`.
  - Keep layout adjustments disabled unless the final rendered pages demonstrate a specific spacing defect.
  - Delete obsolete experiments once THM-30 is complete so commented code does not accumulate as hidden alternatives.

## Recommended execution order

- [ ] Phase A: Complete THM-01 through THM-09 before substantial prose polishing.
- [ ] Phase B: Complete THM-10 through THM-14 as section-specific technical reviews.
- [ ] Phase C: Complete THM-15 through THM-25 as the compression, figure, and citation pass.
- [ ] Phase D: Complete THM-26 through THM-34 as the final consistency, commented-source cleanup, and rendering pass.

## Deferred decisions

- [ ] Decide whether the chapter should remain one long theory chapter or be split into foundational and transport-model chapters. Retain the current structure unless the final thesis balance strongly favors a split.
- [ ] Decide whether the model-regime overview belongs in the chapter introduction, chapter conclusion, or both in abbreviated form.
- [ ] Decide whether detailed SSET transport cycles belong in theory or should move closer to the single-electron results.

<!-- I would divide the master plan into roughly 10–12 prompts:
Chapter conclusion and regime-selection table: THM-01–02.
Introduction and main transitions: THM-03–04.
Global notation and terminology: THM-05–09.
Normal-state section: THM-11.
Microscopic section: THM-10 plus local cleanup.
Macroscopic section: THM-12.
Mesoscopic section, BdG through ABS.
Mesoscopic section, MAR through PAMAR: THM-13.
Stochastic section, \(P(E)\), DCB, and ICPT.
Stochastic section, SSET and chapter integration: THM-14.
Compression, captions, and citations: THM-15–25.
Final consistency, build, and visual review: THM-26–30. 

Make a plan to complete THM-01 through THM-02 of masterplan-theory.md. Inspect all theory files, resolve the notation and coherent/incoherent model boundaries, update the checklist, build the thesis, and verify the affected pages. Do not broaden the task into prose compression.

Renew the context at section boundaries, not after every prompt.
For this plan, start a fresh context approximately here:
Chapter architecture: THM-01–04  
Global notation: THM-05–09  
Normal-state and microscopic sections  
Macroscopic section  
Mesoscopic section  
Stochastic section  
Global compression, citations, and figures  
Final audit and rendering
-->
