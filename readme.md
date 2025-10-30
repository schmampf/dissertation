# TODO

- fix citation problems
- implement subsubsection


% 3. Methods chapter (toward the end or appendix)

% Put all technical and procedural material that’s shared across experiments here:
% 	•	Cryostat setup, filtering, microwave delivery, calibration.
% 	•	MCBJ fabrication and control.
% 	•	Numerical details (FCS solver, Python/Fortran comparison).
% 	•	Data processing, fitting routines, error analysis.

% Having it after the physics chapters works well in your case because readers already understand why each method matters, and you avoid front-loading technical detail.

4. Suggested full table of contents
	1.	Introduction
	•	Motivation: coherent transport, photon-assisted phenomena.
	•	Overview of thesis & contributions.
	2.	Foundations of Superconducting Transport
	•	BCS, Josephson, Andreev, MAR, Tien–Gordon framework.
	3.	Tunnel-Barrier Junctions under Microwave Irradiation
	•	Theory (Dynes + Tien–Gordon)
	•	Results, fits, base temperature, asymmetries.
	4.	Few-Channel Atomic Contacts
	•	Theory (FCS + modified Tien–Gordon)
	•	Results, pincode determination, PAMAR, simulations.
	5.	High-Transmission Regime
	•	Qualitative theory & open questions
	•	Experimental results: fractional features, incoherent pair tunneling.
	6.	Methods and Experimental Setup
	•	Fabrication, measurement, numerical framework.
	7.	Conclusion & Outlook
	•	Summary table comparing regimes; open theoretical challenges.
Appendices: derivations, extra plots, raw data.

⸻

🪶 5. Rule of thumb

Put general principles once (in Foundations)
and specific equations only when they are used (inside each regime chapter).

That keeps your readers oriented and makes it easy to publish parts later as stand-alone papers.

⸻

Would you like me to make a one-page outline table (chapter × theory × experiment × key figure) that you can drop into your project notebook? It’s a very practical writing map.




    %%% List of tables and figures
        % \addcontentsline{toc}{chapter}{List of Figures}
        % \listoffigures

        % \begingroup
        % \let\clearpage\relax
        % \listoftables
        % \addcontentsline{toc}{chapter}{List of Tables}
        % \endgroup