# Experimental Setup Section TODO

This file tracks the work needed to make `methods/setup.tex` a polished thesis section and a useful technical reference for successors. The goal is not only clean prose, but a section that shows deep ownership of the apparatus, documents the decisions made during this work, and makes the measurement constraints reproducible.

## Overall Goal

- [ ] Make the setup section read as a mature experimental methods section, not as a collection of notes.
- [ ] Make clear which parts are general cryogenic or electrical principles, which parts describe the specific LD400 setup, and which parts were modified during this work.
  -> Started for the cryostat section by tying each cooling principle to an experimental constraint.
- [ ] Preserve practical setup knowledge that would help a successor operate, diagnose, or rebuild the experiment.
  -> Started for the cryostat compressor and cooling-water maintenance history.
- [ ] Connect every technical subsystem to the observables used later in the thesis: `\textit{I--V}`, `\textit{dI--dV}`, conductance maps, microwave response, extracted gaps, switching behavior, and noise limits.
  -> Started for the cryostat section by linking thermalization, radiation, and vibration to spectra and contact stability.
- [ ] Avoid overclaiming. Where performance was optimized empirically, state the criterion and the observed consequence.

## Structure and Narrative

- [ ] Decide the role of each main block: cryostat, MCBJ mechanics, electrical measurement chain, microwave coupling, thermometry, and grounding.
- [ ] Add a short roadmap at the start that explains why this order is used.
- [ ] Make each subsection answer three questions:
  - What does this subsystem do?
  - What constraints does it impose on the measurement?
  - What was specific, changed, or learned during this work?
- [ ] End each major subsection with a short takeaway paragraph.
- [x] Check whether `Electrical Wiring` should be renamed to something broader, for example `Electrical Measurement Environment`, because it includes instrumentation and grounding.
- [x] Check whether the starred subsubsections should become numbered subsubsections so they can be referenced later.
  -> No, it is fine that they are not numbered. I do not want to have four numbered references.

## Labels and Cross-References

- [x] Change `\label{section:setup}` to a thesis-guide-compliant label such as `\label{sec:methods:setup}`.
- [x] Add labels directly after each subsection and important subsubsection.
- [x] Use namespaced labels consistently:
  - `sec:methods:setup`
  - `subsec:setup:cryostat`
  - `subsec:setup:mcbj`
  - `subsec:setup:electrical`
  - `subsec:setup:measurement-circuit`
  - `subsec:setup:dc-filtering`
  - `subsec:setup:ac-cabling`
  - `subsec:setup:grounding`
- [x] Add labels to important equations in the Joule--Thomson section.
- [ ] Check all references use `Sec.~\ref{...}`, `Fig.~\ref{...}`, and `Eq.~\eqref{...}` consistently.
- [x] Replace inconsistent `Figure~...` and `Fig.~...` usage with the thesis convention.

## Cryostat Section

- [x] Decide how much cryogenic textbook explanation belongs here versus in a shortened methods-focused form.
  -> Keep the textbook explanations for pulse-tube cooling, Joule--Thomson cooling, and dilution refrigeration, but make each one end in a specific experimental consequence.
- [x] Trim or tighten pulse-tube theory so it supports the specific experimental constraints: precooling, vibration, thermal anchoring, and maintenance.
  -> First pass done. The section still keeps the physics explanation, but the framing now states why it matters for the LD400 and for transport measurements.
- [ ] Add setup-specific values if available:
  - [x] typical base temperature
    -> Added manufacturer-specified 8 mK expected base temperature and practical installed-experiment values of about 30--40 mK.
  - [x] typical mixing-chamber temperature during measurements
    -> Added practical 30--40 mK range.
  - [x] cooldown time
    -> Added about 36 h to cool to the few-kelvin regime including the magnet, plus about 3 h from 4 K to base.
  - [x] condensation time
    -> Added about 1 h.
  - [x] warm-up time
    -> Added normal Bluefors warm-up script duration of about 2--3 days.
  - [x] still temperature range
    -> Added approximate still temperature of about 1 K.
  - [x] 4 K and 40 K stage temperatures
    -> Added typical 40 K stage near 47 K and 4 K stage near 3.2 K.
  - [x] circulation rate
    -> Not available. Martin Prestel did not record a recoverable circulation-rate value, and the relevant logs are lost. Do not keep this as an open TODO.
  - [x] cooling power or manufacturer specification if relevant
    -> Added manufacturer specifications for LD400: guaranteed 10 mK base temperature, expected 8 mK base temperature, 14 uW at 20 mK, 400 uW at 100 mK, and 575 uW at 120 mK. Explicitly stated that these values were not remeasured for the installed experiment.
- [ ] Clarify how pulse-tube vibrations affect MCBJ stability, contact tuning, or noise.
  -> Added air-spring damping at four cryostat corners, about 6 bar operating pressure, the role of the large suspended mass, contact stability on the scale of weeks, and occasional atomic rearrangements during individual measurements. The local AMI magnet specification gives the 7 T magnet weight as 57 lb, about 26 kg. No reliable total cryostat/support-frame mass was found, so keep the total mass qualitative.
- [x] Resolve `TODO[SETUP-CRYO-01]`: add air-spring operating pressure if available.
- [ ] Resolve `TODO[SETUP-CRYO-02]`: add later heater-power calibration for controlled sample-stage warming if the corresponding measurement data are recovered.
  -> Also connect this later to microwave heating: during microwave irradiation, the sample-stage temperature could rise to several hundred millikelvin, and in extreme cases approach 1 K.
- [x] Check the pulse-tube maintenance footnote for tone and placement. It contains valuable successor knowledge, but it may be too long for a footnote.
  -> Moved the maintenance history into the main text so it reads as practical setup knowledge rather than an oversized footnote.
- [x] Decide whether compressor maintenance, water-cooling issues, adsorber replacement, and contamination risks should move into a short practical paragraph or appendix note.
  -> Kept it in the main cryostat text because it explains real setup reliability limits and successor-relevant diagnostics.
- [x] Verify the Joule--Thomson description for helium mixture operation and avoid implying pure `\textsuperscript{4}He` behavior where the mixture is the relevant working fluid.
  -> First pass done by stating that the plotted `\textsuperscript{4}He` curve illustrates the sign change, while the circulating mixture differs quantitatively.
- [x] Add labels to the Joule--Thomson equations.
- [x] Check whether the phase-diagram caption correctly states the phase separation region.
  -> Corrected to below the phase-separation boundary.
- [x] Connect dilution cooling explicitly to electronic temperature, filter thermalization, and measurement resolution.
- [ ] Add a final cryostat takeaway that states the practical temperature and stability conditions under which the measurements were performed.
  -> Added that the MCBJ could be moved step by step while remaining in the 40 mK regime, so base-temperature stability was not the dominant limitation compared with microwave heating.
  -> Conceptual takeaway added. Practical 30--40 mK range, still about 1 K, cooldown and condensation times, stage temperatures, and week-scale contact stability added.

## MCBJ Mechanics

- [x] Make the MCBJ section read as a controlled mechanical reduction chain from motor rotation to atomic-contact elongation.
  -> First pass done. The section now distinguishes motor/drive motion, differential-screw feed, bending, and atomic-contact response.
- [x] Add or verify the mechanical reduction factor from substrate bending to junction elongation if available.
  -> Not independently calibrated in this work. The text now states this explicitly and treats position operationally through relative displacement and conductance response.
- [x] Add the effective displacement per motor step or per encoder count if available.
  -> Raw encoder units are intentionally not used because the scale is inconvenient and not physically transparent. The text now uses the relative 0--18 displacement coordinate, corresponding to 30 differential-screw revolutions and 3 mm macroscopic screw feed.
- [x] Verify and document the differential screw pitch calculation.
  -> Kept effective feed of 0.1 mm per revolution.
- [ ] Add the total gearbox reduction before and after the modifications.
- [x] Explain why the gearbox redistribution improved reliability using torque, heat load, and failure mode language.
  -> Added failure mode: high torque at the first gearbox plus a blockage lower in the drive train can deform the drive rod or decouple the gearbox from the following shaft.
- [x] State which parts were redesigned during this work and which were inherited from earlier group setups.
  -> Added that the inherited platform is from Fischer/Prestel, while motor mounting and gearbox distribution/replacements were part of this work.
- [x] Rework CAD-style captions into thesis captions that state function and consequence.
- [x] Add practical assembly knowledge for successors:
  - [x] alignment while rotating slowly
  - [x] gearbox cleaning procedure
  - [x] lubricant removal
  - [x] solvent and moisture removal
  - [x] feedthrough lubrication
  - [x] failure signs during cooldown or operation
    -> Added cold-function diagnostic: intentional motion should produce sample-stage heating, sometimes up to about 1 K. Monitor gas-handling pressure because still heat load can cause overpressure.
- [x] Decide whether detailed workshop project numbers belong in captions, footnotes, or an appendix.
  -> Moved workshop project numbers from captions to footnotes.
- [x] Connect the mechanics to observable behavior: stable tunneling traces, contact opening and closing, access to few-channel contacts, and reduced risk of motor failure.
- [x] End with a clear takeaway on reproducibility and stability of contact tuning.
  -> Added final paragraph: after the gearbox distribution, motor mount, slip clutch, feedthrough lubrication, and assembly procedure were revised, the mechanics could be operated reliably during cooldown and at base temperature.

## Measurement Circuit

- [x] Make the biasing discussion maximally clear for readers who need to reconstruct `V_\mathrm{sample}` and `I_\mathrm{sample}`.
  -> Revised standard configuration, current reconstruction, and low-impedance fallback.
- [x] Verify all sign conventions for `V_\mathrm{bias}`, `V_\mathrm{sample}`, `V_\mathrm{ref}`, and `I_\mathrm{sample}`.
  -> Avoided detailed sign discussion in the thesis. Text states that plotted polarity is chosen consistently for each dataset.
- [x] State whether `V_\mathrm{ref}` is measured across one reference resistor or a symmetric pair.
  -> It is measured across one cold reference resistor.
- [x] Verify whether the current formula needs a sign or factor of two depending on the exact measurement configuration.
  -> No factor of two enters because `V_\mathrm{ref}` is measured across one reference resistor.
- [x] Add actual reference resistor values for the standard and low-impedance configurations.
  -> Standard cold reference value added as `R_\mathrm{ref}=101.473\,\mathrm{k}\Omega`; low-impedance cold bias resistors stated as nominally `100\,\Omega` each.
- [x] Add typical ranges of sample resistance for tunnel, few-channel, and low-resistance contact regimes.
  -> Added representative values: about 100 Ohm for a closed break junction, about `R_0=1/G_0\simeq12.9 kOhm` for a few-channel atomic contact, and tunnel regime starting roughly around 100 kOhm or 0.1 `G_0`.
- [x] Add the effective cold series resistance in each configuration.
  -> Standard setup now states about 203 kOhm from the two cold reference resistors alone; low-impedance setup states nominally 100 Ohm on each side in the cold bias line.
- [x] Clarify when the setup behaves voltage-biased, mixed-biased, or current-biased.
  -> Kept `\eta=R_\mathrm{series}/R_\mathrm{sample}` explanation and tied it to MCBJ state.
- [x] Explain how the measured voltage channels are converted into calibrated `\textit{I--V}` and `\textit{dI--dV}` curves.
  -> this goes into seperate section about my digital infrastructure / data threadment / evaluation...
- [x] State the limits of the low-impedance configuration, especially warm Johnson--Nyquist noise and reduced filtering.
  -> Added room-temperature current-readout noise tradeoff.
- [x] Check whether the measurement-schematic figure shows all components needed to understand the equations.
  -> Caption now states that `V_\mathrm{ref}` is measured across one reference resistor and that the standard configuration has symmetric cold reference resistors.

## DC Wiring and Filtering

- [x] List the DC line functions clearly:
  - [x] bias
  - [x] sample-voltage readout
  - [x] reference-voltage readout
  - [x] DC gate
  - [x] thermometer
  - [x] heater
  - [ ] spare or diagnostic lines if used
- [x] Add actual wire types, line counts, and approximate resistances if available.
  -> Added Bluefors commercial cabling from room temperature to base and GVLZ169 low-temperature coaxial cable at base. Line resistances are not known.
- [x] Clarify the path from sample pin heads to coaxial twisted pairs and room-temperature instruments.
  -> Revised to pin headers, 90 um insulated copper wires, conductive silver paste, Bluefors cabling, and GVLZ169 base-stage cable.
- [x] Fix grammar and terminology around pin heads or pin headers.
  -> Use pin headers.
- [x] Add filter inventory:
  - [x] copper-powder filters
  - [x] MFT25 filters
  - [x] warm low-pass filters
    -> goes into intrumentation
  - [ ] any commercial or custom RC filters
- [x] Add known cutoff frequencies, attenuation ranges, or manufacturer specifications where relevant.
  -> Added MFT25 manufacturer specification language already present: strong attenuation above 130 MHz, low capacitance, and good thermal contact. CP-filter attenuation is not measured in this work.
- [x] Distinguish between filtering for electrical noise and thermalization for electronic temperature.
- [x] State which filter performance is measured, specified, inferred from comparable setups, or assumed.
  -> CP-filter performance is inferred from Thalmann/group design. Text now emphasizes low DC resistance and broadband skin-effect loss rather than a specific measured attenuation curve. MFT25 claims are manufacturer specifications.
- [x] Explain why several filtering stages are needed instead of relying on one element.
  -> CP filters act as a low-resistance broadband safety stage against microwave leakage, while MFT25 provides a compact commercial filter-thermalizer stage.
- [x] Connect filtering quality to subgap leakage, gap sharpness, and low-current resolution.
  -> Text connects cold filtering/readout to low electronic temperature and resolving small subgap currents.

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
