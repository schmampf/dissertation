
        % \begin{equation}
        %     V(t) = V_0 + A \cos (2\pi\nu t)\,,
        %     \label{eq:josepson:shapiro-drive}
        % \end{equation}
        %     Same as Equation \ref{eq:pat-sinusoidal}

        % \begin{equation}
        %     \phi(t) = \phi_0 + \nu_0t + (2eA/h\nu) \sin(2\pi\nu t)
        %     \label{eq:josepson:shapiro-phase}
        % \end{equation}

        % \begin{equation}
        %     I_\mathrm{S}(V_0) = \sum_{n=-\infty}^{\infty} (-1)^n J_n\!\left( \frac{2eA}{h\nu}\right) \cdot I_\mathrm{C} \sin(\phi_0 + 2\pi(\nu_0t-n\nu t))\,.
        %     \label{eq:josepson:shapiro-current}
        % \end{equation}

        % Equation~\ref{eq:josepson:shapiro-current} follows directly from expanding the nonlinear phase dependence in Equation~\ref{eq:josepson:shapiro-phase} using the Jacobi–Anger identity. This transformation converts the time-dependent sine term in the Josephson phase into a sum over harmonic components, each weighted by a Bessel function $J_n(a)$. The factor $(-1)^n$ arises from the parity property $J_{-n}(a) = (-1)^n J_n(a)$ and introduces alternating signs between even and odd harmonics. Physically, this expansion reveals that the microwave-driven Josephson current can be understood as a superposition of sidebands corresponding to processes in which $n$ photons are absorbed or emitted. The time-averaged current, given in Equation~\ref{eq:josepson:shapiro-current0}, collects the stationary terms that satisfy $\nu_0 = n\nu$, corresponding to the phase-locking condition responsible for the appearance of Shapiro steps.

        % \begin{equation}
        %     I_\mathrm{S}(V_0) = \sum_{n=-\infty}^{\infty} J_n\!\left( \frac{2eA}{h\nu}\right) \cdot I_0\!\left(V_0 - \frac{n h\nu}{2e}\right)
        %     \label{eq:josepson:shapiro-current0}
        % \end{equation}
        % compare with equation \ref{eq:pat-tien-gordon}.
        
        % The resulting current $I_\mathrm{S}(V_0)$ can be interpreted as a superposition of oscillating components, each corresponding to a tunneling process in which an integer number $n$ of photons is absorbed or emitted from the external microwave field. Each term in the sum oscillates at a frequency $\nu_0 - n\nu$, representing sidebands of the Josephson oscillation. Whenever the condition $\nu_0 = n\nu$ is fulfilled, the corresponding term becomes stationary and contributes a constant component to the time-averaged current. This phase-locking condition gives rise to quantized voltage plateaus, known as Shapiro steps, appearing at discrete voltages 
        % \begin{equation}
        %     V_n = n \, \frac{h\nu}{2e}\,.
        %     \label{eq:josephson:shapiro-step-position}
        % \end{equation}
        % Their amplitudes are weighted by the Bessel functions $J_n(2eA/h\nu)$, which describe the strength of $n$-photon coupling between the Josephson oscillation and the external drive.
        
    \subsection{RCSJ Model}
    \label{subsec:josephson:rcsj}

        The resistively and capacitively shunted junction (RCSJ) model provides a classical description of the dynamics of a Josephson junction under bias. In this equivalent circuit, the junction is represented by a parallel combination of the Josephson element, a normal-state resistance $R$, and a capacitance $C$. The current through the junction can therefore be written as
        \begin{equation}
            I = I_\mathrm{C}\sin\phi + \frac{V}{R} + C\frac{\mathrm{d}V}{\mathrm{d}t}\,,
            \label{eq:josephson:rcsj}
        \end{equation}
        where the first term describes the supercurrent, the second term accounts for quasiparticle tunneling, and the third represents the displacement current through the junction capacitance. Combined with the Josephson relation $\dot{\phi} = 2eV/\hbar$, this model captures the full time-dependent response of the junction.

        The behavior of the junction is governed by the dimensionless damping parameter $\beta_\mathrm{C} = 2eI_\mathrm{C}R^2C/\hbar$, known as the Stewart–McCumber parameter. For $\beta_\mathrm{C} \gg 1$, the junction is underdamped and exhibits hysteretic $I$–$V$ characteristics. In contrast, for $\beta_\mathrm{C} \ll 1$, the junction is overdamped, leading to a smooth and non-hysteretic response. The junction investigated in this work falls within the overdamped regime, where the capacitive term can be neglected and the phase dynamics are strongly damped.

    \subsection{Incoherent Tunneling of Cooper Pairs}
    \label{subsec:josephson:itcp}

    \newpage
    \subsection*{TODO}
    \textbf{
        \begin{itemize}
            \item Josephson IV
        \end{itemize}}
    \newpage


        The resistively and capacitively shunted junction (RCSJ) model provides a classical description of the dynamics of a Josephson junction under bias. In this equivalent circuit, the junction is represented by a parallel combination of the Josephson element, a normal-state resistance $R$, and a capacitance $C$. The current through the junction can therefore be written as
        \begin{equation}
            I = I_{\mathrm{C},0}\sin\phi + \frac{V}{R} + C\frac{\mathrm{d}V}{\mathrm{d}t}\,,
            \label{eq:josephson:rcsj}
        \end{equation}
        where the first term describes the supercurrent, the second term accounts for quasiparticle tunneling, and the third represents the displacement current through the junction capacitance. Combined with the AC Josephson relation, this model captures the full time-dependent response of the junction.

        The behavior of the junction is governed by the dimensionless damping parameter $\beta_\mathrm{C} = 2eI_\mathrm{C}R^2C/\hbar$, known as the Stewart–McCumber parameter. For $\beta_\mathrm{C} \gg 1$, the junction is underdamped and exhibits hysteretic $I$–$V$ characteristics. In contrast, for $\beta_\mathrm{C} \ll 1$, the junction is overdamped, leading to a smooth and non-hysteretic response. The junction investigated in this work falls within the overdamped regime, where the capacitive term can be neglected and the phase dynamics are strongly damped.

    \subsection{Incoherent Tunneling of Cooper Pairs}
    \label{subsec:josephson:itcp}

    \newpage
    \subsection*{TODO}
    \textbf{
        \begin{itemize}
            \item Josephson IV
            \item shapiro IV
            \item rscj shaltplan & washboard potential
        \end{itemize}}
    \newpage
