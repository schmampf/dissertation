# Experimental Setup Section TODO

This file tracks the work needed to make `methods/setup.tex` a polished thesis section and a useful technical reference for successors. The goal is not only clean prose, but a section that shows deep ownership of the apparatus, documents the decisions made during this work, and makes the measurement constraints reproducible.

## Overall Goal

- [ ] Make the setup section read as a mature experimental methods section, not as a collection of notes.
- [ ] Make clear which parts are general cryogenic or electrical principles, which parts describe the specific LD400 setup, and which parts were modified during this work.
- [ ] Preserve practical setup knowledge that would help a successor operate, diagnose, or rebuild the experiment.
- [ ] Connect every technical subsystem to the observables used later in the thesis: `\textit{I--V}`, `\textit{dI--dV}`, conductance maps, microwave response, extracted gaps, switching behavior, and noise limits.
- [ ] Avoid overclaiming. Where performance was optimized empirically, state the criterion and the observed consequence.

## Structure and Narrative

- [ ] Decide the role of each main block: cryostat, MCBJ mechanics, electrical measurement chain, microwave coupling, thermometry, and grounding.
- [ ] Add a short roadmap at the start that explains why this order is used.
- [ ] Make each subsection answer three questions:
  - What does this subsystem do?
  - What constraints does it impose on the measurement?
  - What was specific, changed, or learned during this work?
- [ ] End each major subsection with a short takeaway paragraph.
- [ ] Check whether `Electrical Wiring` should be renamed to something broader, for example `Electrical Measurement Environment`, because it includes instrumentation and grounding.
- [ ] Check whether the starred subsubsections should become numbered subsubsections so they can be referenced later.

## Labels and Cross-References

- [ ] Change `\label{section:setup}` to a thesis-guide-compliant label such as `\label{sec:methods:setup}`.
- [ ] Add labels directly after each subsection and important subsubsection.
- [ ] Use namespaced labels consistently:
  - `sec:methods:setup`
  - `subsec:setup:cryostat`
  - `subsec:setup:mcbj`
  - `subsec:setup:electrical`
  - `subsec:setup:measurement-circuit`
  - `subsec:setup:dc-filtering`
  - `subsec:setup:ac-cabling`
  - `subsec:setup:grounding`
- [ ] Add labels to important equations in the Joule--Thomson section.
- [ ] Check all references use `Sec.~\ref{...}`, `Fig.~\ref{...}`, and `Eq.~\eqref{...}` consistently.
- [ ] Replace inconsistent `Figure~...` and `Fig.~...` usage with the thesis convention.

## Cryostat Section

- [ ] Decide how much cryogenic textbook explanation belongs here versus in a shortened methods-focused form.
- [ ] Trim or tighten pulse-tube theory so it supports the specific experimental constraints: precooling, vibration, thermal anchoring, and maintenance.
- [ ] Add setup-specific values if available:
  - [ ] typical base temperature
  - [ ] typical mixing-chamber temperature during measurements
  - [ ] cooldown time
  - [ ] condensation time
  - [ ] still temperature range
  - [ ] 4 K and 40 K stage temperatures
  - [ ] circulation rate
  - [ ] cooling power or manufacturer specification if relevant
- [ ] Clarify how pulse-tube vibrations affect MCBJ stability, contact tuning, or noise.
- [ ] Check the pulse-tube maintenance footnote for tone and placement. It contains valuable successor knowledge, but it may be too long for a footnote.
- [ ] Decide whether compressor maintenance, water-cooling issues, adsorber replacement, and contamination risks should move into a short practical paragraph or appendix note.
- [ ] Verify the Joule--Thomson description for helium mixture operation and avoid implying pure `\textsuperscript{4}He` behavior where the mixture is the relevant working fluid.
- [ ] Add labels to the Joule--Thomson equations.
- [ ] Check whether the phase-diagram caption correctly states the phase separation region.
- [ ] Connect dilution cooling explicitly to electronic temperature, filter thermalization, and measurement resolution.
- [ ] Add a final cryostat takeaway that states the practical temperature and stability conditions under which the measurements were performed.

## MCBJ Mechanics

- [ ] Make the MCBJ section read as a controlled mechanical reduction chain from motor rotation to atomic-contact elongation.
- [ ] Add or verify the mechanical reduction factor from substrate bending to junction elongation if available.
- [ ] Add the effective displacement per motor step or per encoder count if available.
- [ ] Verify and document the differential screw pitch calculation.
- [ ] Add the total gearbox reduction before and after the modifications.
- [ ] Explain why the gearbox redistribution improved reliability using torque, heat load, and failure mode language.
- [ ] State which parts were redesigned during this work and which were inherited from earlier group setups.
- [ ] Rework CAD-style captions into thesis captions that state function and consequence.
- [ ] Add practical assembly knowledge for successors:
  - [ ] alignment while rotating slowly
  - [ ] gearbox cleaning procedure
  - [ ] lubricant removal
  - [ ] solvent and moisture removal
  - [ ] feedthrough lubrication
  - [ ] failure signs during cooldown or operation
- [ ] Decide whether detailed workshop project numbers belong in captions, footnotes, or an appendix.
- [ ] Connect the mechanics to observable behavior: stable tunneling traces, contact opening and closing, access to few-channel contacts, and reduced risk of motor failure.
- [ ] End with a clear takeaway on reproducibility and stability of contact tuning.

## Measurement Circuit

- [ ] Make the biasing discussion maximally clear for readers who need to reconstruct `V_\mathrm{sample}` and `I_\mathrm{sample}`.
- [ ] Verify all sign conventions for `V_\mathrm{bias}`, `V_\mathrm{sample}`, `V_\mathrm{ref}`, and `I_\mathrm{sample}`.
- [ ] State whether `V_\mathrm{ref}` is measured across one reference resistor or a symmetric pair.
- [ ] Verify whether the current formula needs a sign or factor of two depending on the exact measurement configuration.
- [ ] Add actual reference resistor values for the standard and low-impedance configurations.
- [ ] Add typical ranges of sample resistance for tunnel, few-channel, and low-resistance contact regimes.
- [ ] Add the effective cold series resistance in each configuration.
- [ ] Clarify when the setup behaves voltage-biased, mixed-biased, or current-biased.
- [ ] Explain how the measured voltage channels are converted into calibrated `\textit{I--V}` and `\textit{dI--dV}` curves.
- [ ] State the limits of the low-impedance configuration, especially warm Johnson--Nyquist noise and reduced filtering.
- [ ] Check whether the measurement-schematic figure shows all components needed to understand the equations.

## DC Wiring and Filtering

- [ ] List the DC line functions clearly:
  - [ ] bias
  - [ ] sample-voltage readout
  - [ ] reference-voltage readout
  - [ ] DC gate
  - [ ] thermometer
  - [ ] heater
  - [ ] spare or diagnostic lines if used
- [ ] Add actual wire types, line counts, and approximate resistances if available.
- [ ] Clarify the path from sample pin heads to coaxial twisted pairs and room-temperature instruments.
- [ ] Fix grammar and terminology around pin heads or pin headers.
- [ ] Add filter inventory:
  - [ ] copper-powder filters
  - [ ] MFT25 filters
  - [ ] warm low-pass filters
  - [ ] any commercial or custom RC filters
- [ ] Add known cutoff frequencies, attenuation ranges, or manufacturer specifications where relevant.
- [ ] Distinguish between filtering for electrical noise and thermalization for electronic temperature.
- [ ] State which filter performance is measured, specified, inferred from comparable setups, or assumed.
- [ ] Explain why several filtering stages are needed instead of relying on one element.
- [ ] Connect filtering quality to subgap leakage, gap sharpness, and low-current resolution.

## AC Cabling and Microwave Coupling

- [ ] Clarify the two microwave paths: AC gate and on-chip antenna or stripline.
- [ ] Add the physical routing and thermalization points of each line.
- [ ] Add attenuator values and locations.
- [ ] Add room-temperature source details only insofar as they affect delivered microwave power and synchronization.
- [ ] State known frequency range and usable power range.
- [ ] Explain what is calibrated absolutely, what is calibrated relatively, and what remains unknown at the sample.
- [ ] Use Patrick Raif's result carefully: both lines behave effectively as microwave antennas, but the exact local voltage at the junction is not known.
- [ ] Connect this limitation to the later interpretation of photon-assisted tunneling or microwave-driven features.
- [ ] Add a clear scope paragraph: microwave amplitude at the junction is treated as an effective parameter unless otherwise calibrated.

## Thermometry and Heating

- [ ] Add the thermometer type and location.
- [ ] Add how base-stage temperature was monitored during measurements.
- [ ] State whether the measured thermometer temperature is the same as the electronic temperature, and if not, why not.
- [ ] Add heater purpose and typical use cases:
  - [ ] thermal cycling
  - [ ] recovery after mechanical motion
  - [ ] temperature-dependent checks
- [ ] Explain possible thermal lag or gradients between thermometer, filters, and sample.
- [ ] Connect temperature control to superconducting gap extraction and thermal broadening.

## Instrumentation and Grounding

- [ ] State the practical grounding rule in one crisp sentence.
- [ ] Make clear that the cryostat was used as the local experimental ground reference.
- [ ] Document which instruments were floating, isolated, connected to building ground, or disconnected during sensitive measurements.
- [ ] Add a table or compact list of instruments if the prose becomes too dense.
- [ ] Include interface types only where they matter for grounding or synchronization.
- [ ] Add the empirical optimization criterion: low-frequency noise in the sample- and reference-voltage channels under measurement conditions.
- [ ] Add the measured or representative noise level if available.
- [ ] Clarify the role of USB isolation, USB-over-LAN, GPIB, RS-232, and LAN connections.
- [ ] Explain why the magnet supply and unused devices were disconnected during low-noise spectra.
- [ ] State the limits of the grounding solution: optimized empirically, not a universal ground model.
- [ ] Connect grounding quality to the measured `\textit{I--V}` stability and spectral cleanliness.

## Figures and Captions

- [ ] Check every figure caption against the thesis caption structure:
  - main result or function
  - main mechanism or trend
  - essential conditions or parameters only
- [ ] Replace CAD-project-note captions with explanatory captions.
- [ ] Decide whether workshop project numbers should be preserved in footnotes, appendix text, or omitted.
- [ ] Verify that each figure is referenced before or near where it appears.
- [ ] Check figure placement and wrapping in the compiled PDF.
- [ ] Check whether any wrapfigure causes awkward line breaks or excessive whitespace.
- [ ] Ensure panel labels and caption text match the actual figure content.
- [ ] Verify all imported PGF and `pdf_tex` assets compile cleanly from `thesis.tex`.

## Citations and Evidence

- [ ] Check every general cryogenic explanation has appropriate source support.
- [ ] Cite manufacturer data sheets or manuals where performance specifications are quoted.
- [ ] Cite group theses only where they document inherited setup design or comparable measurements.
- [ ] Keep local claims separate from literature claims.
- [ ] Add citations for filter behavior, MFT25 specifications, and cryogenic wiring practices where needed.
- [ ] Avoid unsupported universal statements about noise, thermalization, or microwave attenuation.

## Language and Style Pass

- [ ] Convert long nested sentences into shorter thesis-style prose.
- [ ] Keep methods/results tense consistent: past tense and `I` for work performed in this project.
- [ ] Use `we` only when referring to collaborative published or group work if that is intentional.
- [ ] Remove double-dash or em-dash punctuation where present.
- [ ] Avoid semicolons.
- [ ] Minimize parentheses.
- [ ] Use consistent terminology:
  - `quasiparticle`
  - `Cooper pair`
  - `normal-state`
  - `weak-coupling`
  - `\textit{I--V}`
  - `\textit{dI--dV}`
  - `Joule--Thomson`
  - `mechanically controllable break junction`
- [ ] Check spelling and grammar line by line.
- [ ] Remove repeated setup introductions that say the same thing in different words.

## Successor Documentation

- [ ] Add practical knowledge that is hard to recover from schematics alone.
- [ ] Document known failure modes:
  - [ ] compressor water leaks
  - [ ] contamination or blockage in mixture handling
  - [ ] feedthrough friction
  - [ ] motor overload
  - [ ] gearbox freezing
  - [ ] ground loops
  - [ ] microwave heating
- [ ] For each failure mode, state the observed symptom and the mitigation if known.
- [ ] Decide which practical details belong in the main methods text and which belong in an appendix.
- [ ] Make sure the final text helps a future student understand why the setup is built this way, not only what components are present.

## Compile and Review

- [ ] Build with `latexmk thesis.tex` from the repository root.
- [ ] Inspect the generated PDF around the setup section.
- [ ] Check figure placement, caption readability, and wrapfigure behavior.
- [ ] Check for overfull boxes and citation warnings related to `methods/setup.tex`.
- [ ] Read the section once as an examiner: does it demonstrate expertise and justify the measurement environment?
- [ ] Read the section once as a successor: could a new student understand what to check, preserve, and avoid?
- [ ] Read the section once as a thesis author: does every paragraph earn its place?

