"""Mesoscopic superconductivity helper functions.

This module collects small numerical utilities used in the mesoscopic chapter
(e.g. temperature-dependent BCS gap, current--phase relations in different
limits, Dorokhov/DMPK averaging for KO-1, and simple Shapiro-step / fractional
Shapiro-step visualizations based on a Fourier-expanded CPR).

Units / conventions
-------------------
- Energies and voltages are typically expressed in meV/mV.
- Frequencies are expressed in GHz unless stated otherwise.
- `G_N` is treated as the dimensionless normal conductance `g = G_N/G_0` in the
  DMPK density and related integrals.

The functions in this file are intended for figure generation and sanity checks
rather than performance-critical production code.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scipy.special import jv

from scipy.constants import e
from scipy.constants import h
from scipy.constants import Boltzmann as k_B

NDArray64: TypeAlias = NDArray[np.float64]

# Useful physical constants in mixed units (SI and eV-based) for convenience.
h_e_Vs: float = h / e
G_0_S: float = 2 * e * e / h
k_B_eV: float = k_B / e

h_e_pVs: float = h_e_Vs * 1e12
G_0_muS: float = G_0_S * 1e6
k_B_meV: float = k_B_eV * 1e3


def get_TC(Delta_meV: float = 0.18) -> float:
    """Return the BCS critical temperature Tc inferred from Δ(0).

    Uses the weak-coupling BCS relation Δ(0) = 1.764 kB Tc.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.

    Returns
    -------
    float
        Critical temperature Tc in kelvin.
    """
    T_C_K = Delta_meV / (1.764 * k_B_meV)
    return T_C_K


def Delta_meV_of_T(Delta_meV: float, T_K: float) -> float:
    """Return the BCS energy gap Δ(T) in meV.

    The gap is approximated by the standard weak-coupling BCS interpolation

    Δ(T) = Δ(0) tanh[1.74 sqrt(T_c/T - 1)],

    with Δ(0)=`Delta_meV` and an effective critical temperature inferred
    from Δ(0) = 1.76 k_B T_c.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    float
        Gap Δ(T) in meV.

    Raises
    ------
    ValueError
        If `T_K` is negative.
    """

    T_C_K = get_TC(Delta_meV=Delta_meV)  # Critical temperature in Kelvin
    if T_K < 0:
        raise ValueError("Temperature (K) must be non-negative.")
    if T_K >= T_C_K:
        # warnings.warn(f"Estimated T_C: {T_C_K:.2f} K", category=UserWarning)
        return 0.0
    elif T_K == 0:
        return Delta_meV
    else:
        # BCS theory: Delta(T) = Delta(0) * tanh(1.74 * sqrt(Tc/T - 1))
        return Delta_meV * np.tanh(1.74 * np.sqrt(T_C_K / T_K - 1))


def get_IC_AB(
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> float:
    """Return the Ambegaokar-Baratoff critical current in normalized units.

    The function returns the dimensionless prefactor used throughout this file,
    i.e. a critical current normalized to (G_N * G_0) * Δ.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    float
        Normalized AB critical current (dimensionless).

    Notes
    -----
    The temperature dependence follows

    I_c(T) R_N = (π Δ(T) / 2e) tanh(Δ(T) / 2k_B T).
    """
    Delta_T_meV = Delta_meV_of_T(Delta_meV=Delta_meV, T_K=T_K)
    I_C_nA = np.pi / 2 * Delta_T_meV / Delta_meV
    if T_K > 0.0:
        I_C_nA *= np.tanh(Delta_T_meV / (2 * k_B_meV * T_K))
    return I_C_nA


def get_IC_AB_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> float:
    """Return the Ambegaokar--Baratoff critical current in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.

    Returns
    -------
    float
        AB critical current in nA.
    """
    IC_AB = get_IC_AB(Delta_meV=Delta_meV, T_K=T_K)
    IC_AB_nA = IC_AB * G_N * G_0_muS * Delta_meV
    return IC_AB_nA


def get_CPR_AB(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> NDArray64:
    """Sinusoidal (tunnel-junction) current-phase relation.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Normalized CPR values I(φ) (dimensionless).
    """
    I_C = get_IC_AB(Delta_meV=Delta_meV, T_K=T_K)
    return I_C * np.sin(phi)


def get_CPR_AB_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """Sinusoidal (tunnel-junction) CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        CPR values in nA.
    """
    I_C_nA = get_IC_AB_nA(Delta_meV=Delta_meV, G_N=G_N, T_K=T_K)
    return I_C_nA * np.sin(phi)


def get_E_ABS(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    tau: float = 1.0,
) -> NDArray64:
    """Return the dimensionless ABS energy for a single channel.

    Uses E(φ)/Δ = sqrt(1 - τ sin²(φ/2)) for a short, symmetric
    contact.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    tau
        Normal-state transmission probability τ ∈ [0,1].

    Returns
    -------
    NDArray64
        Dimensionless ABS energy E(φ)/Δ.
    """
    return np.sqrt(1 - tau * np.sin(phi / 2) ** 2)


def get_E_ABS_meV(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    tau: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> NDArray64:
    """Return the ABS energy E(φ) in meV.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    tau
        Transmission probability τ ∈ [0,1].
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        ABS energy in meV.
    """
    Delta_T_meV = Delta_meV_of_T(Delta_meV=Delta_meV, T_K=T_K)
    E_ABS = get_E_ABS(phi=phi, tau=tau)
    E_ABS_meV = E_ABS * Delta_T_meV
    return E_ABS_meV


def get_CPR_ABS(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """ABS-based CPR for a short single-channel contact (normalized).

    The CPR is obtained from the derivative of the ABS energy with respect to
    phase, optionally including thermal occupation via a tanh factor.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Normalized supercurrent I(φ) (dimensionless).

    Notes
    -----
    This implementation uses a numerical gradient of E(φ). For τ→1 the
    CPR develops a cusp near φ=π; increase the phase grid density if you
    need accurate higher harmonics.
    """
    Delta_T_meV = Delta_meV_of_T(Delta_meV=Delta_meV, T_K=T_K)
    E_abs = get_E_ABS(phi=phi, tau=tau)
    E_abs *= Delta_T_meV / Delta_meV

    I_abs = -2 * np.pi * np.gradient(E_abs, phi) / tau

    if T_K > 0.0:
        I_abs *= np.tanh(E_abs * Delta_meV / (2 * k_B_meV * T_K))
    return I_abs


def get_CPR_ABS_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """ABS-based CPR in nA for a single channel.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        CPR values in nA.
    """
    CPR = get_CPR_ABS(phi=phi, Delta_meV=Delta_meV, tau=tau, T_K=T_K)
    CPR_nA = CPR * tau * G_0_muS * Delta_meV
    return CPR_nA


def get_rho(
    tau: NDArray64 = np.arange(1e-5, 1, 1e-5, dtype=np.float64),
    G_N: float = 1.0,
    eps: float = 1e-8,
) -> NDArray64:
    """Dorokhov/DMPK density of transmission eigenvalues ρ(τ).

    Parameters
    ----------
    tau
        Transmission grid τ ∈ (0,1).
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    eps
        Numerical cutoff used to avoid the integrable endpoint divergences at
        τ→0 and τ→1.

    Returns
    -------
    NDArray64
        Density ρ(τ) such that ∫_0^1 dτ ρ(τ) τ = G_N.
    """
    tau = np.clip(tau, eps, 1 - eps)
    rho = G_N / (2 * tau * np.sqrt(1 - tau))
    return rho


def get_CPR_KO1(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    dtau: int = 1e-5,
) -> NDArray64:
    """KO-1 diffusive short-junction CPR obtained by DMPK averaging.

    The diffusive CPR is computed by averaging the single-channel ABS CPR over
    the Dorokhov/DMPK distribution ρ(τ).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    dtau
        Transmission step used for the numerical integral over τ.

    Returns
    -------
    NDArray64
        Normalized KO-1 CPR (dimensionless).

    Notes
    -----
    The endpoint divergences of ρ(τ) are integrable but require a small
    cutoff; convergence is controlled by `dtau`.
    """

    tau = np.arange(dtau, 1, dtau, dtype=np.float64)
    rho = get_rho(tau=tau)

    # evaluate I(phi, tau) and integrate over tau
    I_tau_phi = np.empty((tau.size, phi.size), dtype=np.float64)
    for i, tau_i in enumerate(tau):
        I_tau_phi[i] = (
            get_CPR_ABS(
                phi=phi,
                Delta_meV=Delta_meV,
                tau=tau_i,
                T_K=T_K,
            )
            * tau_i
        )

    # integral over tau (this is what you were missing)
    I_phi = np.trapezoid(rho[:, None] * I_tau_phi, tau, axis=0)
    return I_phi


def get_CPR_KO1_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: int = 1e-5,
) -> NDArray64:
    """KO-1 diffusive CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.
    dtau
        Transmission step used for the numerical integral over τ.

    Returns
    -------
    NDArray64
        KO-1 CPR values in nA.
    """
    CPR_KO1 = get_CPR_KO1(phi=phi, Delta_meV=Delta_meV, T_K=T_K, dtau=dtau)
    CPR_KO1_nA = CPR_KO1 * G_N * G_0_muS * Delta_meV
    return CPR_KO1_nA


def get_CPR_KO2(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
) -> NDArray64:
    """KO-2 clean short-contact CPR (ballistic limit τ=1).

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        Normalized KO-2 CPR (dimensionless).
    """
    return get_CPR_ABS(
        phi=phi,
        Delta_meV=Delta_meV,
        tau=1.0,
        T_K=T_K,
    )


def get_CPR_KO2_nA(
    phi: NDArray64 = np.linspace(0, 2 * np.pi, 101, dtype=np.float64),
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
) -> NDArray64:
    """KO-2 clean short-contact CPR in nA.

    Parameters
    ----------
    phi
        Phase difference φ in radians.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.

    Returns
    -------
    NDArray64
        KO-2 CPR values in nA.
    """
    CPR_KO2 = get_CPR_KO2(phi=phi, Delta_meV=Delta_meV, T_K=T_K)
    CPR_KO2_nA = CPR_KO2 * G_N * G_0_muS * Delta_meV
    return CPR_KO2_nA


def get_IC_ABS(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Return the ABS critical current (normalized) by maximizing I(φ).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        Normalized critical current (dimensionless).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_CPR_ABS(phi=phi, Delta_meV=Delta_meV, tau=tau, T_K=T_K)
    I_C = np.max(CPR)
    return I_C


def get_IC_ABS_nA(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Return the ABS critical current in nA.

    See `get_IC_ABS` for the definition of the maximization.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        Critical current in nA.
    """
    IC_ABS = get_IC_ABS(
        Delta_meV=Delta_meV,
        tau=tau,
        T_K=T_K,
        n_phi=n_phi,
    )
    IC_ABS_nA = IC_ABS * tau * G_0_muS * Delta_meV
    return IC_ABS_nA


def get_IC_KO1(
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> float:
    """Return the KO-1 critical current (normalized) by maximizing I(φ).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    dtau
        Transmission step used for the DMPK integral.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        Normalized KO-1 critical current (dimensionless).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_CPR_KO1(phi=phi, Delta_meV=Delta_meV, T_K=T_K, dtau=dtau)
    I_C = np.max(CPR)
    return I_C


def get_IC_KO1_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    dtau: float = 1e-4,
    n_phi: int = 501,
) -> float:
    """Return the KO-1 critical current in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.
    dtau
        Transmission step used for the DMPK integral.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        KO-1 critical current in nA.
    """
    IC_KO1 = get_IC_KO1(
        Delta_meV=Delta_meV,
        T_K=T_K,
        dtau=dtau,
        n_phi=n_phi,
    )
    IC_KO1_nA = IC_KO1 * G_N * G_0_muS * Delta_meV
    return IC_KO1_nA


def get_IC_KO2(
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Return the KO-2 critical current (normalized) by maximizing I(φ).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        Normalized KO-2 critical current (dimensionless).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    CPR = get_CPR_KO2(phi=phi, Delta_meV=Delta_meV, T_K=T_K)
    I_C = np.max(CPR)
    return I_C


def get_IC_KO2_nA(
    Delta_meV: float = 0.18,
    G_N: float = 1.0,
    T_K: float = 0.0,
    n_phi: int = 501,
) -> float:
    """Return the KO-2 critical current in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    T_K
        Temperature in kelvin.
    n_phi
        Number of phase points used for the maximization.

    Returns
    -------
    float
        KO-2 critical current in nA.
    """
    IC_KO2 = get_IC_KO2(
        Delta_meV=Delta_meV,
        T_K=T_K,
        n_phi=n_phi,
    )
    IC_KO2_nA = IC_KO2 * G_N * G_0_muS * Delta_meV
    return IC_KO2_nA


def fourier_sine_coeffs(
    phi: NDArray64,
    I_phi: NDArray64,
    p_max: int = 10,
):
    """Compute sine-series Fourier coefficients for a 2π-periodic CPR.

    Parameters
    ----------
    phi
        Uniform phase grid on [0, 2π) (use `endpoint=False`).
    I_phi
        CPR samples I(φ) on the same grid.
    p_max
        Maximum harmonic order p to compute.

    Returns
    -------
    NDArray64
        Array of coefficients `I_p` such that
        I(φ) ≈ ∑_{p=1}^{p_{max}} I_p sin(pφ).

    Notes
    -----
    This routine uses direct numerical projection. For strongly skewed CPRs
    (e.g. τ→1 with a cusp near φ=π) increase the grid density.
    """
    # spacing (uniform)
    dphi = phi[1] - phi[0]

    coeffs = np.empty(p_max, dtype=np.float64)
    for p in range(1, p_max + 1):
        coeffs[p - 1] = (1 / np.pi) * np.sum(I_phi * np.sin(p * phi)) * dphi
    return coeffs


def get_I_P_ABS(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    p_max: int = 10,
) -> NDArray64:
    """Return the harmonic amplitudes I_p for an ABS CPR (normalized).

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    p_max
        Maximum harmonic order p.

    Returns
    -------
    NDArray64
        Harmonic amplitudes I_p (dimensionless).
    """
    phi = np.linspace(0, 2 * np.pi, 1001, endpoint=False)
    I_phi = get_CPR_ABS(phi=phi, Delta_meV=Delta_meV, tau=tau, T_K=T_K)
    I_p = fourier_sine_coeffs(phi=phi, I_phi=I_phi, p_max=p_max)
    I_p *= tau
    return I_p


def get_I_P_ABS_nA(
    Delta_meV: float = 0.18,
    tau: float = 1.0,
    T_K: float = 0.0,
    p_max: int = 10,
) -> NDArray64:
    """Return the harmonic amplitudes I_p for an ABS CPR in nA.

    Parameters
    ----------
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    tau
        Transmission probability τ ∈ [0,1].
    T_K
        Temperature in kelvin.
    p_max
        Maximum harmonic order p.

    Returns
    -------
    NDArray64
        Harmonic amplitudes I_p in nA.
    """
    I_p_ABS = get_I_P_ABS(
        Delta_meV=Delta_meV,
        tau=tau,
        T_K=T_K,
        p_max=p_max,
    )
    I_p_ABS_nA = I_p_ABS * tau * G_0_muS * Delta_meV
    return I_p_ABS_nA


def do_I_fSS(
    V_mV: NDArray64,
    A_mV: NDArray64,
    I: NDArray64 = np.array([1.0]),
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Construct a discrete (delta-comb) Shapiro / fractional-Shapiro spectrum.

    This helper places contributions at commensurate voltages

        V_{n/p} = (n/p) (hν) / (2e),

    with weights derived from the CPR harmonics and Bessel functions.

    Parameters
    ----------
    V_mV
        Voltage grid in mV (used only to place peaks onto nearest bins).
    A_mV
        Drive amplitudes V_ac in mV.
    I
        Array of harmonic amplitudes I_p (dimensionless), where index p runs
        from 1..len(I).
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n included in the sum.

    Returns
    -------
    NDArray64
        Array with shape (len(A_mV), len(V_mV)) containing peak amplitudes.

    Notes
    -----
    This is a visualization tool (not a dynamical RCSJ simulation). Peaks are
    accumulated onto the nearest voltage bin.
    """
    I_p = np.copy(I)

    m = 2
    # Photon energy h\nu in meV
    # (nu_GHz -> Hz and eV -> meV conversion combined).
    hnu_meV = h / e * nu_GHz * 1e12
    I_fSS = np.zeros((A_mV.shape[0], V_mV.shape[0]))

    a = np.arange(0, A_mV.shape[0], 1)
    p = np.arange(1, len(I_p) + 1, 1)
    n = np.arange(0, n_max + 1, 1)
    for i_a, _ in enumerate(a):
        for _, n_i in enumerate(n):
            for i_p, p_i in enumerate(p):
                V_np_mV = n_i / p_i * hnu_meV / m
                alpha_p = 2 * p_i * A_mV[i_a] / hnu_meV
                if np.abs(V_np_mV) <= np.nanmax(np.abs(V_mV)):
                    i_V = [
                        np.argmin(np.abs(V_mV - V_np_mV)),
                        np.argmin(np.abs(V_mV + V_np_mV)),
                    ]
                    J_np = jv(n_i, alpha_p)
                    I_np = np.abs(J_np) * np.abs(I_p[i_p])
                    I_fSS[i_a, i_V] += I_np
    return I_fSS


def get_I_SS(
    V_mV: NDArray64,
    A_mV: NDArray64,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Integer Shapiro spectrum for a sinusoidal tunnel-junction CPR.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.

    Returns
    -------
    NDArray64
        Dimensionless peak amplitudes on the voltage grid.
    """
    I_C = get_IC_AB(Delta_meV=Delta_meV, T_K=T_K)
    I_SS = do_I_fSS(V_mV=V_mV, A_mV=A_mV, I=[I_C], nu_GHz=nu_GHz, n_max=n_max)
    return I_SS


def get_I_SS_nA(
    V_mV: NDArray64,
    A_mV: NDArray64,
    G_N: float = 1.0,
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
) -> NDArray64:
    """Integer Shapiro spectrum in nA.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    G_N
        Dimensionless normal conductance `g = G_N/G_0`.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.

    Returns
    -------
    NDArray64
        Peak amplitudes in nA on the voltage grid.
    """
    I_C = get_IC_AB(Delta_meV=Delta_meV, T_K=T_K)
    I_SS = do_I_fSS(V_mV=V_mV, A_mV=A_mV, I=[I_C], nu_GHz=nu_GHz, n_max=n_max)
    I_SS_nA = I_SS * G_N * G_0_muS * Delta_meV
    return I_SS_nA


def get_I_fSS(
    V_mV: NDArray64,
    A_mV: NDArray64,
    tau: NDArray64 = np.array([1.0]),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
    p_max: int = 10,
) -> NDArray64:
    """Fractional Shapiro spectrum from a Fourier-expanded ABS CPR.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    tau
        Iterable of channel transmissions used to build the CPR.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.
    p_max
        Maximum CPR harmonic order p included.

    Returns
    -------
    NDArray64
        Dimensionless peak amplitudes on the voltage grid.

    Notes
    -----
    The CPR harmonics are obtained by projecting the ABS CPR onto sine
    harmonics. The resulting peak weights scale with |I_p J_n(p a)|.
    """
    phi = np.linspace(0, 2 * np.pi, 1001, endpoint=False)
    I_p = np.zeros((p_max))

    for tau_i in tau:
        I_phi = get_CPR_ABS(
            phi=phi,
            Delta_meV=Delta_meV,
            tau=tau_i,
            T_K=T_K,
        )
        I_p += (
            fourier_sine_coeffs(
                phi=phi,
                I_phi=I_phi,
                p_max=p_max,
            )
            * tau_i
        )
    I_p = np.abs(I_p)
    I_fSS = do_I_fSS(
        V_mV=V_mV,
        A_mV=A_mV,
        I=I_p,
        nu_GHz=nu_GHz,
        n_max=n_max,
    )
    return I_fSS


def get_I_fSS_nA(
    V_mV: NDArray64,
    A_mV: NDArray64,
    tau: NDArray64 = np.array([1.0]),
    Delta_meV: float = 0.18,
    T_K: float = 0.0,
    nu_GHz: float = 10.0,
    n_max: int = 1000,
    p_max: int = 10,
) -> NDArray64:
    """Fractional Shapiro spectrum in nA.

    Parameters
    ----------
    V_mV
        Voltage grid in mV.
    A_mV
        Drive amplitudes V_ac in mV.
    tau
        Iterable of channel transmissions used to build the CPR.
    Delta_meV
        Zero-temperature gap Δ(0) in meV.
    T_K
        Temperature in kelvin.
    nu_GHz
        Drive frequency in GHz.
    n_max
        Maximum photon index n.
    p_max
        Maximum CPR harmonic order p included.

    Returns
    -------
    NDArray64
        Peak amplitudes in nA on the voltage grid.
    """
    I_fSS = get_I_fSS(
        V_mV=V_mV,
        A_mV=A_mV,
        tau=tau,
        Delta_meV=Delta_meV,
        T_K=T_K,
        nu_GHz=nu_GHz,
        n_max=n_max,
        p_max=p_max,
    )
    I_fSS_nA = I_fSS * G_0_muS * Delta_meV
    return I_fSS_nA
