# Experimental Setup Section TODO

This file tracks the work needed to make `methods/setup.tex` a polished thesis section and a useful technical reference for successors. The goal is not only clean prose, but a section that shows deep ownership of the apparatus, documents the decisions made during this work, and makes the measurement constraints reproducible.

## Overall Goal

- [ ] Make the setup section read as a mature experimental methods section, not as a collection of notes.
  -> Substantial narrative pass done. The remaining subsections now end with measurement consequences rather than component inventory.
- [ ] Make clear which parts are general cryogenic or electrical principles, which parts describe the specific LD400 setup, and which parts were modified during this work.
  -> Started for the cryostat section by tying each cooling principle to an experimental constraint. Extended to the electrical subsections by separating measurement schematic, cryostat wiring, filtering, thermometry, microwave coupling, and room-temperature grounding.
- [ ] Preserve practical setup knowledge that would help a successor operate, diagnose, or rebuild the experiment.
  -> Started for the cryostat compressor and cooling-water maintenance history. Extended to reference-resistor redesign, filter-stack reliability, thermometry cable separation, microwave-line limitations, and grounding diagnostics.
- [ ] Connect every technical subsystem to the observables used later in the thesis: `\textit{I--V}`, `\textit{dI--dV}`, conductance maps, microwave response, extracted gaps, switching behavior, and noise limits.
  -> Started for the cryostat section by linking thermalization, radiation, and vibration to spectra and contact stability. Extended to the electrical subsections by stating how voltage/current reconstruction, filtering, temperature readout, microwave coupling, and grounding constrain the later spectra and maps.
- [ ] Avoid overclaiming. Where performance was optimized empirically, state the criterion and the observed consequence.
  -> Added explicit scope language for electronic temperature, microwave amplitude, filter performance, and empirical grounding optimization.

## Structure and Narrative

- [x] Decide the role of each main block: cryostat, MCBJ mechanics, electrical measurement chain, microwave coupling, thermometry, and grounding.
  -> Added roadmap framing by experimental constraint: cryogenic operating window, mechanical contact tuning, then electrical interpretation environment.
- [x] Add a short roadmap at the start that explains why this order is used.
  -> Added an explicit roadmap after the setup overview.
- [x] Make each subsection answer three questions:
  - What does this subsystem do?
  - What constraints does it impose on the measurement?
  - What was specific, changed, or learned during this work?
  -> Cryogenic environment, MCBJ mechanics, and electrical measurement environment now each state function, measurement constraint, and project-specific lessons.
- [x] End each major subsection with a short takeaway paragraph.
  -> Cryogenic, MCBJ, and electrical sections now each close with an interpretation/reproducibility takeaway.
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
- [x] Check all references use `Sec.~\ref{...}`, `Fig.~\ref{...}`, and `Eq.~\eqref{...}` consistently.
  -> Use full `Figure~\ref{...}` at the beginning of a sentence and abbreviated `Fig.~\ref{...}` inside a sentence. Keep `Sec.~\ref{...}` and `Eq.~\eqref{...}` for section and equation references.
- [x] Replace inconsistent figure-reference usage with the thesis convention.
  -> Do not start a sentence with `Fig.~...`.

## Citation Pass

- [x] Add local references for sources that should not be stored only in the Zotero export.
  -> Added local entries for the Cryomech PT415/CP1110 manual, Basel Precision Instruments MFT product page, and the Johnson/Nyquist thermal-noise papers.
- [x] Cite manufacturer/specification claims locally.
  -> Added citations for the compressor purification chain and the MFT25 manufacturer attenuation/thermalization specification.
- [x] Cite non-obvious noise and microwave conversion claims.
  -> Added Johnson/Nyquist citations for warm reference-resistor noise and cold-reference-resistor relocation. Added a microwave-engineering citation for the 10 dBm to 1 V matched-50 Ohm conversion.
- [ ] Check later whether part-specific footnotes should be converted into bibliography entries where a stable datasheet or manual exists.
  -> Keep workshop project numbers and one-off component identifiers in footnotes. Use bibliography entries for sources that support physical claims or manufacturer specifications.

## Cryostat Section

- [x] Decide how much cryogenic textbook explanation belongs here versus in a shortened methods-focused form.
  -> Keep the textbook explanations for pulse-tube cooling, Joule--Thomson cooling, and dilution refrigeration, but make each one end in a specific experimental consequence.
- [x] Trim or tighten pulse-tube theory so it supports the specific experimental constraints: precooling, vibration, thermal anchoring, and maintenance.
  -> First pass done. The section still keeps the physics explanation, but the framing now states why it matters for the LD400 and for transport measurements.
- [x] Add setup-specific values if available:
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
- [x] Clarify how pulse-tube vibrations affect MCBJ stability, contact tuning, or noise.
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
- [x] Add a final cryostat takeaway that states the practical temperature and stability conditions under which the measurements were performed.
  -> Added that the MCBJ could be moved step by step while remaining in the 40 mK regime, so base-temperature stability was not the dominant limitation compared with microwave heating.
  -> Conceptual takeaway added. Practical 30--40 mK range, still about 1 K, cooldown and condensation times, stage temperatures, and week-scale contact stability added.

## MCBJ Mechanics

- [x] Make the MCBJ section read as a controlled mechanical reduction chain from motor rotation to atomic-contact elongation.
  -> Narrative check done. The section now distinguishes motor/drive motion, differential-screw feed, bending, and atomic-contact response.
- [x] Add or verify the mechanical reduction factor from substrate bending to junction elongation if available.
  -> Not independently calibrated in this work. The text now states this explicitly and treats position operationally through relative displacement and conductance response.
- [x] Add the effective displacement per motor step or per encoder count if available.
  -> Raw encoder units are intentionally not used because the scale is inconvenient and not physically transparent. The text now uses the relative 0--18 displacement coordinate, corresponding to 30 differential-screw revolutions and 3 mm macroscopic screw feed.
  -> Added explicit scope language that the physical calibration comes from the conductance response, not from the encoder number itself.
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
    -> Fomblin vacuum grease used. Exact type remains open as `TODO[SETUP-MCBJ-GREASE]`.
  - [x] failure signs during cooldown or operation
    -> Added cold-function diagnostic: intentional motion should produce sample-stage heating, sometimes up to about 1 K. Monitor gas-handling pressure because still heat load can cause overpressure.
- [x] Decide whether detailed workshop project numbers belong in captions, footnotes, or an appendix.
  -> Moved workshop project numbers from captions to footnotes.
- [x] Connect the mechanics to observable behavior: stable tunneling traces, contact opening and closing, access to few-channel contacts, and reduced risk of motor failure.
- [x] End with a clear takeaway on reproducibility and stability of contact tuning.
  -> Added final paragraph: after the gearbox distribution, motor mount, slip clutch, feedthrough lubrication, and assembly procedure were revised, the mechanics could be operated reliably during cooldown and at base temperature.
- [ ] Resolve `TODO[SETUP-MCBJ-GREASE]`: add the exact Fomblin vacuum-grease type used on the vacuum feedthrough.

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
  - [x] spare or diagnostic lines if used
    -> The DC gate uses two physical lines shorted at the lowest practical point for diagnostic and backup access.
- [x] Add actual wire types, line counts, and approximate resistances if available.
  -> Added Bluefors commercial cabling from room temperature to base and GVLZ169 low-temperature coaxial cable at base, including LEMO connector identifiers and copper bend-protection tubes in a footnote. Line resistances are not known.
- [x] Clarify the path from sample pin headers to coaxial twisted pairs and room-temperature instruments.
  -> Revised to pin headers, 90 um insulated copper wires, conductive silver paste, Bluefors cabling, and GVLZ169 base-stage cable.
- [x] Fix grammar and terminology around pin heads or pin headers.
  -> Use pin headers.
- [x] Add filter inventory:
  - [x] copper-powder filters
  - [x] MFT25 filters
  - [x] warm low-pass filters
    -> goes into intrumentation
  - [x] any commercial or custom RC filters
    -> No additional custom RC filters are documented for the cryostat wiring. The warm Thorlabs EF120 low-pass filter is described in the instrumentation path.
- [x] Add known cutoff frequencies, attenuation ranges, or manufacturer specifications where relevant.
  -> Added MFT25 manufacturer specification language already present: strong attenuation above 130 MHz, low capacitance, and good thermal contact. Added MFT25 part/serial and Micro-D connector information in a footnote. CP-filter attenuation is not measured in this work.
- [x] Distinguish between filtering for electrical noise and thermalization for electronic temperature.
- [x] State which filter performance is measured, specified, inferred from comparable setups, or assumed.
  -> CP-filter performance is inferred from Thalmann/group design. Text now emphasizes low DC resistance and broadband skin-effect loss rather than a specific measured attenuation curve. MFT25 claims are manufacturer specifications.
- [x] Explain why several filtering stages are needed instead of relying on one element.
  -> CP filters act as a low-resistance broadband safety stage against microwave leakage, while MFT25 provides a compact commercial filter-thermalizer stage.
- [x] Connect filtering quality to subgap leakage, gap sharpness, and low-current resolution.
  -> Text connects cold filtering/readout to low electronic temperature and resolving small subgap currents.
- [ ] Resolve `TODO[SETUP-REF-HOUSING]`: add the scientific-workshop project number or identifier for the reference-resistor housing after it can be checked.

## Thermometry and Heating

- [x] Add the thermometer type and location.
  -> Location added: mounted on the sample holder next to the sample, following the earlier MCBJ setup used by Martin Prestel. Sensor added as Rox resistor thermometer RX102B-CB. Serial number `U06030` and curve name `MCBJ Sample ORI` moved to a footnote.
- [x] Add how base-stage temperature was monitored during measurements.
  -> Added local sample-holder thermometer readout and separate thermometer/heater wiring through a separate cable tree and break-out box, with the Bluefors-cabling LEMO plug identifier in a footnote.
- [x] State whether the measured thermometer temperature is the same as the electronic temperature, and if not, why not.
  -> Added distinction between local sample-holder temperature and junction electronic temperature.
- [x] Add heater purpose and typical use cases:
  - [x] thermal cycling
  - [x] recovery after mechanical motion
  - [x] temperature-dependent checks
  -> Added about 100 Ohm insulated manganin heater wire wrapped around a sample-holder support post.
- [x] Explain possible thermal lag or gradients between thermometer, filters, and sample.
  -> Added that imperfect line thermalization, residual radiation, bias dissipation, and microwave excitation can make the electronic temperature exceed the thermometer reading.
- [x] Connect temperature control to superconducting gap extraction and thermal broadening.
  -> Added that temperature-dependent SIS spectra can check electronic temperature through thermal broadening, but this belongs to data analysis rather than direct thermometry.

## AC Cabling and Microwave Coupling

- [x] Clarify the two microwave paths: AC gate and on-chip antenna or stripline.
  -> Use `AC-gate line` for the on-chip capacitive gate and `antenna line` for the stripped coaxial cable near the sample.
- [x] Add the physical routing and thermalization points of each line.
  -> Added coaxial microwave routing, superconducting commercial cryostat cabling, pass-through connectors at each temperature stage, and base-temperature cabling installed together with Patrick Raif.
- [x] Add attenuator values and locations.
  -> Added 10 dB attenuators at the 4 K flange and pass-through connectors at the other stages. Added future recommendation to move attenuation to the MXC pass-through.
- [x] Add room-temperature source details only insofar as they affect delivered microwave power and synchronization.
  -> later in instrumentation.
- [x] State known frequency range and usable power range.
  -> Added reliable transmission up to about 20 GHz based on microwave-induced spectral changes. Maximum applied AC source power was 10 dBm, corresponding to 1 V peak amplitude in a matched 50 Ohm system. This is not the local sample voltage.
- [x] Explain what is calibrated absolutely, what is calibrated relatively, and what remains unknown at the sample.
  -> Added that the sample coupling is not impedance matched and that absolute local microwave-voltage calibration was not attempted. Calibration is instead effective and uses internal spectral references where available, with the SIS photon-assisted-tunneling cross-reference given once in the AC-cabling text. Relative transmission differences and spectral responses are meaningful, while absolute microwave voltage at the junction remains unknown.
- [x] Use Patrick Raif's result carefully: both lines behave effectively as microwave antennas, but the exact local voltage at the junction is not known.
- [x] Connect this limitation to the later interpretation of photon-assisted tunneling or microwave-driven features.
- [x] Add a clear scope paragraph: microwave amplitude at the junction is treated as an effective parameter unless otherwise calibrated.

## Instrumentation and Grounding

- [x] State the practical grounding rule in one crisp sentence.
  -> Added: instruments connect to the cryostat reference only through intended signal paths or deliberately isolated control links.
- [x] Make clear that the cryostat was used as the local experimental ground reference.
- [x] Document which instruments were floating, isolated, connected to building ground, or disconnected during sensitive measurements.
  -> Added isolated USB/USB-over-LAN control chain, LAN-connected instruments, floating motor controller/LAN switch, and disconnected magnet/diagnostic devices.
- [x] Add a table or compact list of instruments if the prose becomes too dense.
  -> Kept prose grouped by function. No table needed yet.
- [x] Include interface types only where they matter for grounding or synchronization.
  -> Kept GPIB, RS-232, USB, LAN, measurement-card synchronization, and Lake Shore 372AC continuous sample-thermometer readout where they affect grounding, isolation, paired voltage readout, or PID temperature control. Added thermometry break-out box `WW-2480259` for separating the sample line from auxiliary thermometry lines.
- [x] Add the empirical optimization criterion: low-frequency noise in the sample- and reference-voltage channels under measurement conditions.
- [ ] Add the measured or representative noise level if available.
  -> Still open until a representative measured value is available.
- [x] Clarify the role of USB isolation, USB-over-LAN, GPIB, RS-232, and LAN connections.
  -> Added National Instruments GPIB-USB-HS, Icron USB 2.0 Ranger 2304, CESYS C028149, and D-Link DGS-108 identifiers.
- [x] Explain why the magnet supply and unused devices were disconnected during low-noise spectra.
- [x] State the limits of the grounding solution: optimized empirically, not a universal ground model.
- [x] Connect grounding quality to the measured `\textit{I--V}` stability and spectral cleanliness.

## Figures and Captions

- [ ] Check every figure caption against the thesis caption structure:
  - main result or function
  - main mechanism or trend
  - essential conditions or parameters only
- [ ] Replace CAD-project-note captions with explanatory captions.
- [x] Decide whether workshop project numbers should be preserved in footnotes, appendix text, or omitted.
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

- [x] Convert long nested sentences into shorter thesis-style prose.
  -> First pass done in cryostat operation, dilution cooling, wiring, filtering, microwave coupling, biasing, and instrumentation paragraphs. Remaining long physical lines are mainly captions or footnote artifacts rather than nested prose.
- [x] Keep methods/results tense consistent: past tense and `I` for work performed in this project.
  -> Current first-person usage is for work performed in this project.
- [x] Use `we` only when referring to collaborative published or group work if that is intentional.
  -> Current `we` usage refers to the collaborative result with Patrick Raif.
- [x] Remove double-dash or em-dash punctuation where present.
  -> No em dashes found. Remaining double hyphens are intentional LaTeX notation or terms such as `\textit{I--V}` and `Joule--Thomson`.
- [x] Avoid semicolons.
  -> No semicolons found in `methods/setup.tex`.
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

- [x] Add practical knowledge that is hard to recover from schematics alone.
  -> Added compressor maintenance, water-cooling issues, vacuum and cooldown diagnostics, gearbox cleaning, feedthrough lubrication, wiring continuity checks, grounding isolation, and microwave-heating limits.
- [x] Document known failure modes:
  - [x] compressor water leaks
  - [x] contamination or blockage in mixture handling
  - [x] feedthrough friction
  - [x] motor overload
  - [x] gearbox freezing
  - [x] ground loops
  - [x] microwave heating
- [x] For each failure mode, state the observed symptom and the mitigation if known.
  -> Current text states the observed consequence and practical mitigation for each known failure mode at methods-section level.
- [x] Decide which practical details belong in the main methods text and which belong in an appendix.
  -> Kept details in the main methods text when they affect reliability, temperature, noise, or reproducible operation. Exact missing part identifiers remain TODOs rather than appendix material.
- [x] Make sure the final text helps a future student understand why the setup is built this way, not only what components are present.
  -> Current section emphasizes practical constraints, failure modes, and measurement consequences rather than only listing parts.

## Compile and Review

- [ ] Build with `latexmk thesis.tex` from the repository root.
- [ ] Inspect the generated PDF around the setup section.
- [ ] Check figure placement, caption readability, and wrapfigure behavior.
- [ ] Check for overfull boxes and citation warnings related to `methods/setup.tex`.
- [ ] Read the section once as an examiner: does it demonstrate expertise and justify the measurement environment?
- [ ] Read the section once as a successor: could a new student understand what to check, preserve, and avoid?
- [ ] Read the section once as a thesis author: does every paragraph earn its place?
