# Stochastic Section Correction Checklist

This checklist tracks Elke's handwritten corrections in `Theory and Methods_ES_v3.pdf`, PDF pages 49--63, corresponding to printed thesis pages 47--61. We will address the items one at a time in `theory/stochastic.tex`.

## Working method

- [ ] Discuss the correction and agree on its intended meaning.
- [ ] Edit the corresponding passage in `theory/stochastic.tex`.
- [ ] Check the physical argument, notation, references, and connection to observables.
- [ ] Compile `thesis.tex` and inspect the affected pages.
- [ ] Mark the item complete here.

## 1. Opening and meaning of the stochastic description

- [x] **STOCH-01, PDF p. 49, printed p. 47:** Rename the section from `Stochastic Description` to `Stochastic Transport`.
- [x] **STOCH-02, PDF p. 49, printed p. 47:** Replace the vague pronoun in "It produces voltage fluctuations" and identify the fluctuating junction voltage explicitly.
- [x] **STOCH-03, PDF p. 49, printed p. 47:** Explain that the stochastic elements are the environmental voltage and phase fluctuations, exchanged energy, and probabilistic tunneling events.
- [x] **STOCH-04, PDF p. 49, printed p. 47:** Define $\delta V(t)$ relative to the mean bias and distinguish the phase acquired by a single-electron tunneling amplitude from the condensate phase difference.

## 2. General \(P(E)\) framework

- [x] **STOCH-05, PDF p. 51, printed p. 49:** Define $q$ using concrete single-electron and Cooper-pair examples and replace the abstract "selection rules" with the initial and final electronic states of the process.
- [x] **STOCH-06, PDF p. 51, printed p. 49:** Show explicitly that $P_q(E)=\delta(E)$ gives $\Gamma_q^{\rightarrow}(V)=F_q^{\rightarrow}(0,V)$ and link the single-electron case to the microscopic tunnel rate.
- [x] **STOCH-07, PDF p. 52, printed p. 50:** Introduce the cutoff energy and Euler Gamma function directly after Eq. (108).
- [x] **STOCH-08, PDF p. 52, printed p. 50:** Remove the distracting factorial analogy and notation warning while retaining the normalization role of the Euler Gamma function.
- [x] **STOCH-09, PDF p. 53, printed p. 51:** Explain how the three environmental kernels in Fig. 28 produce observable suppression, broadening, and satellites after convolution with the electronic kernel.
- [x] **STOCH-10, PDF p. 53, printed p. 51:** Connect transferred charge, the $q^2$ dependence of the phase correlator, the charge-specific $P_q(E)$ kernels, transition rates, and SSET occupations.

## 3. Dynamical Coulomb blockade

- [x] **STOCH-11, PDF p. 53, printed p. 51:** Explain Fig. 29 through the energy balance between the bias, quasiparticle excitations, and the environment, and expand its caption.
- [x] **STOCH-12, PDF p. 53, printed p. 51:** Explain that restricted low-energy exchange suppresses the tunneling rate and therefore reduces the zero-bias slope of the current.
- [x] **STOCH-13, PDF p. 54, printed p. 52:** Derive the squared Bessel weights conceptually from the AC phase, photon-order amplitudes, and time averaging.
- [x] **STOCH-14, PDF p. 55, printed p. 53:** Replace "current-replica formulation" and spell out incoherent Cooper-pair tunneling before its dedicated subsection.
- [x] **STOCH-15, PDF p. 56, printed p. 54:** Revise the paired captions to contrast Poisson-weighted passive resonances with squared Bessel-weighted driven replicas.

## 4. DCB of multiple Andreev reflection

- [x] **STOCH-16, PDF pp. 57--58, printed pp. 55--56:** Distinguish experiments on island Coulomb blockade competing with MAR from the predicted series-impedance DCB correction, and avoid an absolute claim that DCB of MAR has never been measured.

## 5. Incoherent Cooper-pair tunneling

- [ ] **STOCH-17, PDF p. 58, printed p. 56:** Introduce the abbreviation ICPT directly in the subsection title or first sentence.
- [ ] **STOCH-18, PDF p. 58, printed p. 56:** Add the missing conceptual explanation before the ICPT rate equations. State why coherent Josephson transport is lost while incoherent pair transfer remains possible.
- [ ] **STOCH-19, PDF p. 59, printed p. 57:** Reconsider or remove the paragraph on incoherent Andreev reflection. Elke crossed out the statement that it is mentioned only for completeness.
- [ ] **STOCH-20, PDF p. 59, printed p. 57:** Check that the final ICPT takeaway clearly distinguishes coherent Shapiro locking from probabilistic photon assisted pair transfer.

## 6. Conventional SET and SSET foundations

- [ ] **STOCH-21, PDF p. 60, printed p. 58:** Change "conventional SET" to "normal conducting SET" where that is the intended contrast with the SSET.
- [ ] **STOCH-22, PDF p. 60, printed p. 58:** Clarify whether odd island charge states need to be considered already in the normal state discussion. Elke's bottom note appears to ask "consider odd \(n\)?".
- [ ] **STOCH-23, PDF p. 61, printed p. 59:** Rewrite the transition from the normal conducting SET to the SSET so that it is clear what changes when both leads and the island become superconducting.
- [ ] **STOCH-24, PDF p. 61, printed p. 59:** Explain how the superconducting gap changes the allowed transitions and the voltage thresholds. Elke's long blue margin note is only partly legible, so its precise wording should be confirmed together.
- [ ] **STOCH-25, PDF p. 61, printed p. 59:** Revise the parity paragraph. State carefully how odd charge states acquire an additional free energy and under which conditions the periodicity changes between \(e\) and \(2e\).

## 7. JQP cycle and master equation

- [ ] **STOCH-26, PDF p. 61, printed p. 59:** Correct the description of the JQP cycle. A Cooper pair changes the island charge by two elementary charges, and the full cycle contains two quasiparticle tunneling steps that restore the initial charge state.
- [ ] **STOCH-27, PDF p. 61, printed p. 59:** Explain the order and junction assignment of the pair and quasiparticle transitions. Relate the JQP onset to the relevant bias and gate thresholds. This is the main substantive correction on the page.
- [ ] **STOCH-28, PDF p. 61, printed p. 59:** Reassess the claim that treating the resonant pair step as an incoherent rate is "controlled" under the stated condition. Define the relevant dephasing or lifetime broadening scale and compare it correctly with \(E_J/\hbar\).
- [ ] **STOCH-29, PDF p. 61, printed p. 59:** Clarify when the population-only orthodox master equation is sufficient and when off-diagonal density matrix elements require a generalized master equation.
- [ ] **STOCH-30, PDF pp. 61--62, printed pp. 59--60:** Check the transition-rate sign convention and the statement about positive and negative available free energy. Ensure it is consistent with the definitions used in the equations.

## 8. Tunable and hybrid SSETs

- [ ] **STOCH-31, PDF p. 62, printed p. 60:** Replace "the microscopic MAR mechanism is developed in Sec. 1.4.4" with "is described" or equivalent.
- [ ] **STOCH-32, PDF p. 62, printed p. 60:** Improve the sentence ending with "coherent multiparticle transfer". Elke appears to suggest adding "though" or otherwise making the limitation more explicit.
- [ ] **STOCH-33, PDF p. 63, printed p. 61:** Rewrite "the charge-state picture remains useful but is not sufficient by itself" as a clearer statement that it cannot fully describe the strong coupling regime.
- [ ] **STOCH-34, PDF p. 63, printed p. 61:** Check the final takeaway paragraph against the revised scope: population master equations for separable stochastic events, generalized master equations for charge-state coherence, and breakdown of sharply defined island charge at strong coupling.

## Notes requiring confirmation

- [x] Resolve the section title on printed p. 47 as `Stochastic Transport`.
- [x] Address the short margin note beside the definition of $q$ by defining the net transferred charge explicitly and explaining its role in the $q^2$ environmental coupling.
- [ ] Confirm the long blue margin note concerning superconducting transitions and voltage thresholds on printed p. 59.
- [ ] Confirm the red and green comments surrounding the JQP and incoherent-rate paragraphs on printed p. 59.
