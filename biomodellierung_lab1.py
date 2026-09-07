#!/usr/bin/env python3
"""Complete Python solution for Lab 1: membrane potential and ion channels.

The script implements
  1. the GHK voltage and GHK current equations,
  2. a spherical single-compartment model,
  3. stochastic voltage-dependent m and h gates,
  4. a stochastic m^3 h sodium channel and Na spikes.

All internal calculations use SI units. Voltages are stored in volts,
times in seconds, and currents in amperes. The concentration values in the
table can be used directly because 1 mM = 1 mol/m^3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Constants and laboratory data
# -----------------------------------------------------------------------------

R = 8.314  # J / (K mol), value from the assignment
F = 96480.0  # C / mol, value from the assignment
TEMPERATURE = 293.0  # K

ION_NAMES = np.array(["K", "Na", "Cl"])
Z = np.array([1.0, 1.0, -1.0])
P_BASE = np.array([4.00e-9, 0.12e-9, 0.40e-9])  # m/s
C_IN = np.array([400.0, 50.0, 40.0])  # mM = mol/m^3
C_OUT = np.array([10.0, 460.0, 5.0])  # mM = mol/m^3

SPECIFIC_CAPACITANCE = 0.01  # F/m^2 = 1 microF/cm^2
DT = 0.1e-3  # s
T_END = 50e-3  # s
SOMA_DIAMETER = 100e-6  # m
SPINE_DIAMETER = 1e-6  # m
SINGLE_NA_CONDUCTANCE = 1e-12  # S

DEFAULT_SEED = 12345
DEFAULT_STIMULUS_PA = 0.200


# -----------------------------------------------------------------------------
# GHK equations
# -----------------------------------------------------------------------------

def _x_over_one_minus_exp_minus_x(x: np.ndarray | float) -> np.ndarray:
    """Stable evaluation of x / (1 - exp(-x)), including x near zero."""

    x_arr = np.asarray(x, dtype=float)
    small = np.abs(x_arr) < 1e-7
    result = np.ones_like(x_arr)
    np.divide(
        x_arr,
        -np.expm1(-x_arr),
        out=result,
        where=~small,
    )
    series = 1.0 + x_arr / 2.0 + x_arr**2 / 12.0 - x_arr**4 / 720.0
    return np.where(small, series, result)


def ghk_voltage(
    permeability: np.ndarray = P_BASE,
    c_in: np.ndarray = C_IN,
    c_out: np.ndarray = C_OUT,
) -> float:
    """GHK resting potential for K, Na, and Cl in volts.

    For the anion Cl, the intracellular and extracellular concentrations
    are interchanged in the GHK voltage equation.
    """

    p = np.asarray(permeability, dtype=float)
    numerator = p[0] * c_out[0] + p[1] * c_out[1] + p[2] * c_in[2]
    denominator = p[0] * c_in[0] + p[1] * c_in[1] + p[2] * c_out[2]
    if numerator <= 0.0 or denominator <= 0.0:
        raise ValueError("GHK voltage requires positive numerator and denominator.")
    return float(R * TEMPERATURE / F * np.log(numerator / denominator))


def nernst_potentials(
    c_in: np.ndarray = C_IN,
    c_out: np.ndarray = C_OUT,
) -> np.ndarray:
    """Nernst potentials for each ion species in volts."""

    return R * TEMPERATURE / (Z * F) * np.log(c_out / c_in)


def ghk_current_density_outward(
    voltage: np.ndarray | float,
    permeability: np.ndarray = P_BASE,
    c_in: np.ndarray = C_IN,
    c_out: np.ndarray = C_OUT,
) -> np.ndarray:
    """GHK current density for each ion species, positive outward, in A/m^2.

    The last axis of the result corresponds to [K, Na, Cl]. At V=0, the
    correct limiting value P*z*F*(C_in-C_out) is used automatically.
    """

    v = np.asarray(voltage, dtype=float)
    xi = v[..., np.newaxis] * Z * F / (R * TEMPERATURE)
    factor = _x_over_one_minus_exp_minus_x(xi)
    return (
        np.asarray(permeability)
        * Z
        * F
        * factor
        * (np.asarray(c_in) - np.asarray(c_out) * np.exp(-xi))
    )


def ghk_total_current_density_outward(
    voltage: np.ndarray | float,
    permeability: np.ndarray = P_BASE,
) -> np.ndarray:
    """Sum of the GHK current densities, positive outward, in A/m^2."""

    return np.sum(
        ghk_current_density_outward(voltage, permeability), axis=-1
    )


def sphere_area(diameter: float) -> float:
    """Surface area of a sphere from its diameter: 4*pi*r^2 = pi*d^2."""

    return float(np.pi * diameter**2)


def capacitance(diameter: float) -> float:
    """Total capacitance of a spherical compartment in farads."""

    return SPECIFIC_CAPACITANCE * sphere_area(diameter)


def slope_conductance_density(voltage: float, dv: float = 1e-7) -> float:
    """Differential membrane conductance dJ_out/dV in S/m^2."""

    upper = ghk_total_current_density_outward(voltage + dv)
    lower = ghk_total_current_density_outward(voltage - dv)
    return float((upper - lower) / (2.0 * dv))


# -----------------------------------------------------------------------------
# Passive spherical compartment
# -----------------------------------------------------------------------------

def protocol_permeabilities(time: float) -> np.ndarray:
    """Time protocol from Section 4 of the assignment."""

    p = P_BASE.copy()
    if 10e-3 <= time < 15e-3:
        p[1] = 6.00e-9
    if 25e-3 <= time < 30e-3:
        p[0] = 40.0e-9
    return p


def simulate_passive_compartment(
    diameter: float,
    *,
    changing_permeabilities: bool,
    initial_voltage: float = -50e-3,
    dt: float = DT,
    t_end: float = T_END,
) -> dict[str, np.ndarray | float]:
    """Forward Euler simulation of a passive membrane compartment."""

    time = np.arange(0.0, t_end + 0.5 * dt, dt)
    voltage = np.empty(time.size)
    current_inward = np.empty(time.size)
    voltage[0] = initial_voltage

    area = sphere_area(diameter)
    cm_total = capacitance(diameter)

    for k in range(time.size - 1):
        p = (
            protocol_permeabilities(float(time[k]))
            if changing_permeabilities
            else P_BASE
        )
        # GHK is positive outward; the ODE requires positive inward current.
        current_inward[k] = -area * ghk_total_current_density_outward(
            voltage[k], p
        )
        voltage[k + 1] = (
            voltage[k] + current_inward[k] / cm_total * dt
        )

    p_last = (
        protocol_permeabilities(float(time[-1]))
        if changing_permeabilities
        else P_BASE
    )
    current_inward[-1] = -area * ghk_total_current_density_outward(
        voltage[-1], p_last
    )
    return {
        "time": time,
        "voltage": voltage,
        "current_inward": current_inward,
        "area": area,
        "capacitance": cm_total,
    }


def estimate_passive_time_constant(
    time: np.ndarray, voltage: np.ndarray, target_voltage: float
) -> tuple[float, float]:
    """Return the 1/e estimate and exponential-fit estimate of tau in seconds."""

    ratio = (voltage - target_voltage) / (voltage[0] - target_voltage)
    idx = int(np.argmin(np.abs(ratio - np.exp(-1.0))))
    tau_one_over_e = float(time[idx])

    mask = (time > 0.0) & (time <= 40e-3) & (ratio > 0.0)
    slope, _ = np.polyfit(time[mask], np.log(ratio[mask]), 1)
    tau_fit = float(-1.0 / slope)
    return tau_one_over_e, tau_fit


# -----------------------------------------------------------------------------
# Voltage-dependent stochastic gates
# -----------------------------------------------------------------------------

def m_rates(voltage: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Opening and closing rates of the Na-m gate in 1/s."""

    v = np.asarray(voltage, dtype=float)
    x = (v + 0.035) / 0.010
    # This reformulation avoids the removable singularity at V=-35 mV.
    alpha = 1000.0 * _x_over_one_minus_exp_minus_x(x)
    beta = 4000.0 * np.exp(-(v + 0.060) / 0.018)
    return alpha, beta


def h_rates(voltage: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Opening and closing rates of the Na-h gate in 1/s."""

    v = np.asarray(voltage, dtype=float)
    alpha = 12.0 * np.exp(-v / 0.020)
    beta = 180.0 / (np.exp(-(v + 0.030) / 0.010) + 1.0)
    return alpha, beta


def steady_state_and_tau(
    alpha: np.ndarray | float, beta: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Steady-state open probability and relaxation time."""

    rate_sum = np.asarray(alpha) + np.asarray(beta)
    return np.asarray(alpha) / rate_sum, 1.0 / rate_sum


def transition_probabilities(
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    dt: float,
    *,
    method: str = "exact",
) -> tuple[np.ndarray, np.ndarray]:
    """P(C->O) and P(O->C) over one time step.

    exact integrates the two-state Markov process exactly if the rates
    remain constant within the step. euler reproduces alpha*dt and
    beta*dt from the assignment, but rejects invalid probabilities.
    """

    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    if method == "exact":
        rate_sum = a + b
        switched = -np.expm1(-rate_sum * dt)
        p_open = a / rate_sum * switched
        p_close = b / rate_sum * switched
    elif method == "euler":
        p_open = a * dt
        p_close = b * dt
        if np.any(p_open > 1.0) or np.any(p_close > 1.0):
            raise ValueError(
                "alpha*dt or beta*dt exceeds 1; reduce dt or use method='exact'."
            )
    else:
        raise ValueError("method must be 'exact' or 'euler'.")
    return p_open, p_close


def update_gate(
    state: np.ndarray | bool,
    alpha: np.ndarray | float,
    beta: np.ndarray | float,
    dt: float,
    random_numbers: np.ndarray | float,
) -> np.ndarray:
    """One exact time step for independent two-state gates."""

    p_open, p_close = transition_probabilities(alpha, beta, dt, method="exact")
    current = np.asarray(state, dtype=bool)
    u = np.asarray(random_numbers)
    return np.where(current, ~(u < p_close), u < p_open)


def simulate_m_gate_traces(
    voltages: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
    dt: float = DT,
    t_end: float = T_END,
) -> tuple[np.ndarray, np.ndarray]:
    """A single m gate at each constant holding voltage."""

    time = np.arange(0.0, t_end + 0.5 * dt, dt)
    traces = np.zeros((len(voltages), len(time)), dtype=np.int8)

    for i, voltage in enumerate(voltages):
        rng = np.random.default_rng(seed + i)
        alpha, beta = m_rates(voltage)
        state = False
        for k in range(len(time) - 1):
            state = bool(
                update_gate(state, alpha, beta, dt, rng.random())
            )
            traces[i, k + 1] = state
    return time, traces


def simulate_m3h_channel_traces(
    voltages: np.ndarray,
    *,
    seed: int = DEFAULT_SEED + 1000,
    dt: float = DT,
    t_end: float = T_END,
) -> tuple[np.ndarray, np.ndarray]:
    """One m^3h sodium channel at each constant holding voltage."""

    time = np.arange(0.0, t_end + 0.5 * dt, dt)
    traces = np.zeros((len(voltages), len(time)), dtype=np.int8)

    for i, voltage in enumerate(voltages):
        rng = np.random.default_rng(seed + i)
        m_state = np.zeros(3, dtype=bool)
        h_state = False
        alpha_m, beta_m = m_rates(voltage)
        alpha_h, beta_h = h_rates(voltage)
        for k in range(len(time) - 1):
            m_state = update_gate(
                m_state, alpha_m, beta_m, dt, rng.random(3)
            )
            h_state = bool(
                update_gate(h_state, alpha_h, beta_h, dt, rng.random())
            )
            traces[i, k + 1] = bool(np.all(m_state) and h_state)
    return time, traces


# -----------------------------------------------------------------------------
# Spine compartment with stochastic Na channels
# -----------------------------------------------------------------------------

def make_channel_noise(
    seed: int,
    n_steps: int,
    max_channels: int,
) -> np.ndarray:
    """Pre-generated random numbers for reproducible channel comparisons."""

    rng = np.random.default_rng(seed)
    return rng.random((n_steps - 1, 4, max_channels))


def simulate_spine_with_na_channels(
    stimulus_pA: float,
    n_channels: int,
    *,
    seed: int = DEFAULT_SEED,
    k_pulse: bool = False,
    shared_noise: np.ndarray | None = None,
    dt: float = DT,
    t_end: float = T_END,
) -> dict[str, np.ndarray]:
    """Stochastic spine model from Section 6.

    The stimulus is active between 10 and 15 ms. Optionally, P_K is
    increased tenfold between 20 and 25 ms. As in the example code,
    all gates start in the closed state.
    """

    if n_channels < 0:
        raise ValueError("n_channels must be non-negative.")
    time = np.arange(0.0, t_end + 0.5 * dt, dt)
    voltage = np.empty(time.size)
    voltage[0] = ghk_voltage()
    open_channels = np.zeros(time.size, dtype=int)
    total_inward_current = np.zeros(time.size)

    area = sphere_area(SPINE_DIAMETER)
    cm_total = capacitance(SPINE_DIAMETER)
    e_na = nernst_potentials()[1]

    m_state = np.zeros((3, n_channels), dtype=bool)
    h_state = np.zeros(n_channels, dtype=bool)
    if shared_noise is None:
        shared_noise = make_channel_noise(
            seed, time.size, max(n_channels, 1)
        )
    if shared_noise.shape[0] < time.size - 1 or shared_noise.shape[2] < n_channels:
        raise ValueError("shared_noise is too small for this simulation.")

    for k in range(time.size - 1):
        if n_channels:
            alpha_m, beta_m = m_rates(voltage[k])
            alpha_h, beta_h = h_rates(voltage[k])
            for gate in range(3):
                m_state[gate] = update_gate(
                    m_state[gate],
                    alpha_m,
                    beta_m,
                    dt,
                    shared_noise[k, gate, :n_channels],
                )
            h_state = update_gate(
                h_state,
                alpha_h,
                beta_h,
                dt,
                shared_noise[k, 3, :n_channels],
            )
            open_channels[k] = int(
                np.count_nonzero(np.all(m_state, axis=0) & h_state)
            )

        permeability = P_BASE.copy()
        if k_pulse and 20e-3 <= time[k] < 25e-3:
            permeability[0] *= 10.0

        leak_inward = -area * ghk_total_current_density_outward(
            voltage[k], permeability
        )
        active_na_inward = (
            open_channels[k]
            * SINGLE_NA_CONDUCTANCE
            * (e_na - voltage[k])
        )
        injected = (
            stimulus_pA * 1e-12
            if 10e-3 <= time[k] < 15e-3
            else 0.0
        )
        total_inward_current[k] = leak_inward + active_na_inward + injected
        voltage[k + 1] = (
            voltage[k] + total_inward_current[k] / cm_total * dt
        )

    open_channels[-1] = open_channels[-2]
    total_inward_current[-1] = total_inward_current[-2]
    return {
        "time": time,
        "voltage": voltage,
        "open_channels": open_channels,
        "total_inward_current": total_inward_current,
    }


def simulate_spike_ensemble(
    stimulus_pA: float,
    *,
    n_trials: int,
    n_channels: int = 40,
    seed: int = DEFAULT_SEED + 5000,
    dt: float = DT,
    t_end: float = T_END,
) -> np.ndarray:
    """Vectorized Monte Carlo simulation; return the peak voltages."""

    rng = np.random.default_rng(seed)
    time = np.arange(0.0, t_end + 0.5 * dt, dt)
    voltage = np.full(n_trials, ghk_voltage())
    peak_voltage = voltage.copy()
    m_state = np.zeros((n_trials, 3, n_channels), dtype=bool)
    h_state = np.zeros((n_trials, n_channels), dtype=bool)

    area = sphere_area(SPINE_DIAMETER)
    cm_total = capacitance(SPINE_DIAMETER)
    e_na = nernst_potentials()[1]

    for k in range(time.size - 1):
        alpha_m, beta_m = m_rates(voltage)
        alpha_h, beta_h = h_rates(voltage)
        p_open_m, p_close_m = transition_probabilities(
            alpha_m, beta_m, dt
        )
        p_open_h, p_close_h = transition_probabilities(
            alpha_h, beta_h, dt
        )

        for gate in range(3):
            u = rng.random((n_trials, n_channels))
            m_state[:, gate, :] = np.where(
                m_state[:, gate, :],
                ~(u < p_close_m[:, np.newaxis]),
                u < p_open_m[:, np.newaxis],
            )
        u = rng.random((n_trials, n_channels))
        h_state = np.where(
            h_state,
            ~(u < p_close_h[:, np.newaxis]),
            u < p_open_h[:, np.newaxis],
        )

        n_open = np.count_nonzero(
            np.all(m_state, axis=1) & h_state, axis=1
        )
        leak_inward = -area * ghk_total_current_density_outward(voltage)
        active_na_inward = (
            n_open * SINGLE_NA_CONDUCTANCE * (e_na - voltage)
        )
        injected = (
            stimulus_pA * 1e-12
            if 10e-3 <= time[k] < 15e-3
            else 0.0
        )
        voltage = voltage + (
            leak_inward + active_na_inward + injected
        ) / cm_total * dt
        peak_voltage = np.maximum(peak_voltage, voltage)

    return peak_voltage


def probability_threshold(
    currents: np.ndarray,
    probabilities: np.ndarray,
    target_probability: float,
) -> float:
    """Linearly interpolate the first crossing of the target probability."""

    indices = np.flatnonzero(probabilities >= target_probability)
    if not len(indices):
        return float("nan")
    i = int(indices[0])
    if i == 0:
        return float(currents[0])
    x0, x1 = currents[i - 1], currents[i]
    y0, y1 = probabilities[i - 1], probabilities[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target_probability - y0) * (x1 - x0) / (y1 - y0))


# -----------------------------------------------------------------------------
# Figures and tabular output
# -----------------------------------------------------------------------------

def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_binary_trace_grid(
    time: np.ndarray,
    traces: np.ndarray,
    voltages: np.ndarray,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(12, 10), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for i, voltage in enumerate(voltages):
        ax = axes_flat[i]
        ax.step(time * 1e3, traces[i], where="post", lw=0.8)
        ax.set_title(f"{voltage * 1e3:.0f} mV", fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
    for ax in axes_flat[len(voltages) :]:
        ax.axis("off")
    fig.supxlabel("Time (ms)")
    fig.supylabel("State (0 = closed, 1 = open)")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def run_all(output_dir: Path, n_trials: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )

    # Section 3: GHK voltage
    v_rest = ghk_voltage()
    v_swapped = ghk_voltage(P_BASE, C_OUT, C_IN)
    only_k = np.array([P_BASE[0], 0.0, 0.0])
    only_na = np.array([0.0, P_BASE[1], 0.0])
    v_k = ghk_voltage(only_k)
    v_na = ghk_voltage(only_na)
    e_ions = nernst_potentials()

    current_at_minus70 = ghk_current_density_outward(-70e-3)
    current_at_zero = ghk_current_density_outward(0.0)

    voltages_iv = np.arange(-80.0, 80.0 + 2.5, 5.0) * 1e-3
    currents_iv = ghk_current_density_outward(voltages_iv)
    total_iv = np.sum(currents_iv, axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, name in enumerate(ION_NAMES):
        ax.plot(voltages_iv * 1e3, currents_iv[:, i], label=str(name))
    ax.plot(voltages_iv * 1e3, total_iv, "k", lw=2.2, label="Total")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.axvline(v_rest * 1e3, color="0.4", ls="--", lw=1.0)
    ax.set(xlabel="Membrane potential (mV)", ylabel="Outward GHK current density (A/m²)")
    ax.set_title("GHK current-voltage relationship")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_ghk_iv.png")
    np.savetxt(
        output_dir / "01_ghk_iv.csv",
        np.column_stack((voltages_iv * 1e3, currents_iv, total_iv)),
        delimiter=",",
        header="V_mV,J_K_A_per_m2,J_Na_A_per_m2,J_Cl_A_per_m2,J_total_A_per_m2",
        comments="",
    )

    # Section 4: passive compartment and conductance
    soma_area = sphere_area(SOMA_DIAMETER)
    spine_area = sphere_area(SPINE_DIAMETER)
    j_minus50 = float(ghk_total_current_density_outward(-50e-3))
    i_soma_in_minus50 = -soma_area * j_minus50
    i_spine_in_minus50 = -spine_area * j_minus50

    slope_density_minus50 = slope_conductance_density(-50e-3)
    slope_g_soma = soma_area * slope_density_minus50
    chord_ion_density = ghk_current_density_outward(-50e-3) / (
        -50e-3 - e_ions
    )
    chord_g_soma = soma_area * float(np.sum(chord_ion_density))
    overall_secant_g_soma = soma_area * j_minus50 / (-50e-3 - v_rest)

    passive_soma = simulate_passive_compartment(
        SOMA_DIAMETER, changing_permeabilities=False
    )
    tau_one_e, tau_fit = estimate_passive_time_constant(
        np.asarray(passive_soma["time"]),
        np.asarray(passive_soma["voltage"]),
        v_rest,
    )
    time = np.asarray(passive_soma["time"])
    voltage = np.asarray(passive_soma["voltage"])
    exponential_fit = v_rest + (voltage[0] - v_rest) * np.exp(-time / tau_fit)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(time * 1e3, voltage * 1e3, label="Euler simulation")
    ax.plot(time * 1e3, exponential_fit * 1e3, "--", label=f"Exponential fit: τ = {tau_fit * 1e3:.2f} ms")
    ax.axhline(v_rest * 1e3, color="0.4", ls=":", label="GHK resting potential")
    ax.set(xlabel="Time (ms)", ylabel="Membrane potential (mV)")
    ax.set_title("Passive soma starting at -50 mV")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "02_passive_soma.png")

    protocol_soma = simulate_passive_compartment(
        SOMA_DIAMETER, changing_permeabilities=True
    )
    protocol_spine = simulate_passive_compartment(
        SPINE_DIAMETER, changing_permeabilities=True
    )
    protocol_time = np.asarray(protocol_soma["time"])
    target = np.array(
        [ghk_voltage(protocol_permeabilities(float(t))) for t in protocol_time]
    )
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(protocol_time * 1e3, np.asarray(protocol_soma["voltage"]) * 1e3, label="Soma (100 µm)")
    axes[0].plot(protocol_time * 1e3, np.asarray(protocol_spine["voltage"]) * 1e3, "--", label="Spine (1 µm)")
    axes[0].plot(protocol_time * 1e3, target * 1e3, ":", color="0.35", label="instantaneous GHK target")
    axes[0].set_ylabel("V (mV)")
    axes[0].legend(ncol=2)
    axes[0].set_title("Time-dependent Na and K permeability")
    axes[1].plot(protocol_time * 1e3, np.asarray(protocol_soma["current_inward"]) * 1e12)
    axes[1].set_ylabel("Soma current (pA)\ninward")
    axes[2].plot(protocol_time * 1e3, np.asarray(protocol_spine["current_inward"]) * 1e12)
    axes[2].set_ylabel("Spine current (pA)\ninward")
    axes[2].set_xlabel("Time (ms)")
    for ax in axes:
        for boundary in (10, 15, 25, 30):
            ax.axvline(boundary, color="0.8", lw=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, "03_permeability_protocol_and_size.png")

    protocol_v = np.asarray(protocol_soma["voltage"])
    peak_protocol_idx = int(np.argmax(protocol_v))
    min_protocol_idx = int(np.argmin(protocol_v))

    # Section 5: gate kinetics and individual traces
    voltage_steps = np.arange(-80.0, 80.0 + 5.0, 10.0) * 1e-3
    voltage_dense = np.linspace(-80e-3, 80e-3, 801)
    alpha_m_dense, beta_m_dense = m_rates(voltage_dense)
    m_inf_dense, tau_m_dense = steady_state_and_tau(alpha_m_dense, beta_m_dense)
    alpha_m_steps, beta_m_steps = m_rates(voltage_steps)
    m_inf_steps, tau_m_steps = steady_state_and_tau(alpha_m_steps, beta_m_steps)
    alpha_h_steps, beta_h_steps = h_rates(voltage_steps)
    h_inf_steps, _ = steady_state_and_tau(alpha_h_steps, beta_h_steps)
    channel_open_probability = m_inf_steps**3 * h_inf_steps

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(voltage_dense * 1e3, m_inf_dense)
    axes[0].plot(voltage_steps * 1e3, m_inf_steps, "o", ms=4)
    axes[0].set(xlabel="V (mV)", ylabel="m∞", title="Steady-state activation")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].plot(voltage_dense * 1e3, tau_m_dense * 1e3)
    axes[1].plot(voltage_steps * 1e3, tau_m_steps * 1e3, "o", ms=4)
    axes[1].set(xlabel="V (mV)", ylabel="τm (ms)", title="Time constant of the m gate")
    fig.tight_layout()
    save_figure(fig, output_dir, "04_m_gate_kinetics.png")
    np.savetxt(
        output_dir / "04_gate_kinetics.csv",
        np.column_stack(
            (
                voltage_steps * 1e3,
                alpha_m_steps,
                beta_m_steps,
                m_inf_steps,
                tau_m_steps * 1e3,
                h_inf_steps,
                channel_open_probability,
            )
        ),
        delimiter=",",
        header="V_mV,alpha_m_per_s,beta_m_per_s,m_inf,tau_m_ms,h_inf,m3h_open_probability",
        comments="",
    )

    gate_time, m_traces = simulate_m_gate_traces(voltage_steps)
    plot_binary_trace_grid(
        gate_time,
        m_traces,
        voltage_steps,
        "Stochastic state of a single Na-m gate",
        output_dir,
        "05_m_gate_traces.png",
    )
    channel_time, channel_traces = simulate_m3h_channel_traces(voltage_steps)
    plot_binary_trace_grid(
        channel_time,
        channel_traces,
        voltage_steps,
        "Stochastic state of an m³h sodium channel",
        output_dir,
        "06_m3h_channel_traces.png",
    )

    # Section 6: Na spike, channel count, and K pulse
    n_steps = int(round(T_END / DT)) + 1
    shared_noise = make_channel_noise(DEFAULT_SEED, n_steps, 40)
    active_trace = simulate_spine_with_na_channels(
        DEFAULT_STIMULUS_PA, 40, shared_noise=shared_noise
    )
    passive_trace = simulate_spine_with_na_channels(
        DEFAULT_STIMULUS_PA, 0, shared_noise=shared_noise
    )
    spike_time = np.asarray(active_trace["time"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(spike_time * 1e3, np.asarray(active_trace["voltage"]) * 1e3, label="40 Na channels")
    axes[0].plot(spike_time * 1e3, np.asarray(passive_trace["voltage"]) * 1e3, "--", label="0 Na channels")
    axes[0].axhline(0.0, color="0.5", lw=0.7)
    axes[0].set_ylabel("V (mV)")
    axes[0].legend()
    axes[0].set_title(f"Spine with a {DEFAULT_STIMULUS_PA:.3f} pA stimulus current")
    axes[1].step(spike_time * 1e3, active_trace["open_channels"], where="post")
    axes[1].set(xlabel="Time (ms)", ylabel="open channels")
    for ax in axes:
        ax.axvspan(10, 15, color="tab:orange", alpha=0.15, label="Stimulus")
    fig.tight_layout()
    save_figure(fig, output_dir, "07_spike_active_vs_passive.png")

    channel_numbers = np.array([0, 10, 20, 30, 40])
    channel_traces: dict[int, dict[str, np.ndarray]] = {}
    peaks_by_channel = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for n_channels in channel_numbers:
        result = simulate_spine_with_na_channels(
            DEFAULT_STIMULUS_PA,
            int(n_channels),
            shared_noise=shared_noise,
        )
        channel_traces[int(n_channels)] = result
        peak_mv = float(np.max(result["voltage"]) * 1e3)
        peaks_by_channel.append(peak_mv)
        axes[0].plot(
            result["time"] * 1e3,
            result["voltage"] * 1e3,
            label=f"N = {n_channels}",
        )
    axes[0].axvspan(10, 15, color="tab:orange", alpha=0.15)
    axes[0].set(xlabel="Time (ms)", ylabel="V (mV)", title="Individual traces")
    axes[0].legend(ncol=2)
    axes[1].plot(channel_numbers, peaks_by_channel, "o-")
    axes[1].set(xlabel="Number of Na channels", ylabel="Peak potential (mV)", title="Nonlinear threshold")
    fig.tight_layout()
    save_figure(fig, output_dir, "08_channel_number_effect.png")

    no_k_pulse = active_trace
    with_k_pulse = simulate_spine_with_na_channels(
        DEFAULT_STIMULUS_PA,
        40,
        shared_noise=shared_noise,
        k_pulse=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(spike_time * 1e3, no_k_pulse["voltage"] * 1e3, label="normal P_K")
    ax.plot(spike_time * 1e3, with_k_pulse["voltage"] * 1e3, label="10 × P_K at 20-25 ms")
    ax.axvspan(10, 15, color="tab:orange", alpha=0.12, label="Stimulus")
    ax.axvspan(20, 25, color="tab:blue", alpha=0.10, label="K pulse")
    ax.set(xlabel="Time (ms)", ylabel="V (mV)", title="Effect of increased K permeability")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "09_k_permeability_pulse.png")

    currents_probability = np.arange(0.140, 0.250 + 0.0025, 0.005)
    spike_probabilities = np.empty_like(currents_probability)
    median_peaks = np.empty_like(currents_probability)
    for i, current in enumerate(currents_probability):
        ensemble_peaks = simulate_spike_ensemble(
            float(current), n_trials=n_trials
        )
        # Explicit assumption: a spike occurs if V_peak >= 0 mV.
        spike_probabilities[i] = np.mean(ensemble_peaks >= 0.0)
        median_peaks[i] = np.median(ensemble_peaks) * 1e3

    i10 = probability_threshold(currents_probability, spike_probabilities, 0.10)
    i50 = probability_threshold(currents_probability, spike_probabilities, 0.50)
    i90 = probability_threshold(currents_probability, spike_probabilities, 0.90)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(currents_probability, spike_probabilities, "o-")
    ax.axhline(0.5, color="0.5", ls="--")
    if np.isfinite(i50):
        ax.axvline(i50, color="tab:red", ls="--", label=f"I50 ≈ {i50:.3f} pA")
    ax.set(
        xlabel="Stimulus current (pA; 10-15 ms)",
        ylabel="P(Vpeak ≥ 0 mV)",
        title=f"Stochastic spike probability ({n_trials} trials)",
        ylim=(-0.03, 1.03),
    )
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "10_spike_probability.png")
    np.savetxt(
        output_dir / "10_spike_probability.csv",
        np.column_stack((currents_probability, spike_probabilities, median_peaks)),
        delimiter=",",
        header="stimulus_pA,spike_probability_Vpeak_ge_0,median_peak_mV",
        comments="",
    )

    fixed_seed_thresholds = []
    for current in np.arange(0.150, 0.221, 0.001):
        trace = simulate_spine_with_na_channels(
            float(current), 40, shared_noise=shared_noise
        )
        if float(np.max(trace["voltage"])) >= 0.0:
            fixed_seed_thresholds.append(float(current))
    first_fixed_seed_crossing = (
        min(fixed_seed_thresholds) if fixed_seed_thresholds else float("nan")
    )

    summary: dict[str, object] = {
        "assumptions": {
            "temperature_K": TEMPERATURE,
            "ghk_current_sign": "positive outward",
            "compartment_current_sign": "positive inward",
            "concentrations": "table values used as mol/m^3 because 1 mM = 1 mol/m^3",
            "sphere_area": "pi*d^2",
            "gate_initial_state": "all gates closed, matching the supplied Matlab example",
            "gate_step": "exact two-state Markov transition for constant rates within dt",
            "spike_definition": "peak membrane potential >= 0 mV",
            "random_seed_single_traces": DEFAULT_SEED,
        },
        "section_3": {
            "resting_potential_mV": v_rest * 1e3,
            "swapped_concentrations_mV": v_swapped * 1e3,
            "K_only_mV": v_k * 1e3,
            "Na_only_mV": v_na * 1e3,
            "nernst_potentials_mV": dict(zip(ION_NAMES.tolist(), (e_ions * 1e3).tolist())),
            "J_outward_at_minus70_A_per_m2_by_ion": dict(zip(ION_NAMES.tolist(), current_at_minus70.tolist())),
            "J_outward_at_minus70_total_A_per_m2": float(np.sum(current_at_minus70)),
            "J_outward_at_zero_A_per_m2_by_ion": dict(zip(ION_NAMES.tolist(), current_at_zero.tolist())),
            "J_outward_at_zero_total_A_per_m2": float(np.sum(current_at_zero)),
        },
        "section_4": {
            "soma_area_m2": soma_area,
            "soma_capacitance_F": capacitance(SOMA_DIAMETER),
            "spine_area_m2": spine_area,
            "spine_capacitance_F": capacitance(SPINE_DIAMETER),
            "J_outward_at_minus50_A_per_m2": j_minus50,
            "I_soma_inward_at_minus50_A": i_soma_in_minus50,
            "I_spine_inward_at_minus50_A": i_spine_in_minus50,
            "slope_conductance_soma_at_minus50_S": slope_g_soma,
            "ion_chord_conductance_soma_at_minus50_S": chord_g_soma,
            "overall_secant_conductance_soma_at_minus50_S": overall_secant_g_soma,
            "tau_one_over_e_ms": tau_one_e * 1e3,
            "tau_exponential_fit_ms": tau_fit * 1e3,
            "protocol_peak_mV": float(protocol_v[peak_protocol_idx] * 1e3),
            "protocol_peak_time_ms": float(protocol_time[peak_protocol_idx] * 1e3),
            "protocol_minimum_mV": float(protocol_v[min_protocol_idx] * 1e3),
            "protocol_minimum_time_ms": float(protocol_time[min_protocol_idx] * 1e3),
            "protocol_final_mV": float(protocol_v[-1] * 1e3),
            "GHK_target_high_Na_mV": ghk_voltage(np.array([4e-9, 6e-9, 0.4e-9])) * 1e3,
            "GHK_target_high_K_mV": ghk_voltage(np.array([40e-9, 0.12e-9, 0.4e-9])) * 1e3,
        },
        "section_5": {
            "max_alpha_or_beta_times_dt_on_requested_voltage_grid": float(
                np.max(np.maximum(alpha_m_steps, beta_m_steps) * DT)
            ),
            "m3h_stationary_open_probability_max": float(np.max(channel_open_probability)),
            "m3h_stationary_open_probability_max_voltage_mV": float(
                voltage_steps[np.argmax(channel_open_probability)] * 1e3
            ),
        },
        "section_6": {
            "demonstration_stimulus_pA": DEFAULT_STIMULUS_PA,
            "fixed_seed_active_40_peak_mV": float(np.max(active_trace["voltage"]) * 1e3),
            "fixed_seed_passive_peak_mV": float(np.max(passive_trace["voltage"]) * 1e3),
            "fixed_seed_first_crossing_on_0p001_pA_grid": first_fixed_seed_crossing,
            "peak_mV_by_channel_number": dict(zip(channel_numbers.astype(str).tolist(), peaks_by_channel)),
            "K_pulse_peak_mV": float(np.max(with_k_pulse["voltage"]) * 1e3),
            "K_pulse_voltage_at_25ms_mV": float(with_k_pulse["voltage"][np.argmin(np.abs(spike_time - 25e-3))] * 1e3),
            "monte_carlo_trials_per_current": n_trials,
            "I10_pA": i10,
            "I50_pA": i50,
            "I90_pA": i90,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    s3 = summary["section_3"]
    s4 = summary["section_4"]
    s6 = summary["section_6"]
    print("\nKEY RESULTS")
    print(f"V_rest = {s3['resting_potential_mV']:.3f} mV")
    print(f"Swapped concentrations: {s3['swapped_concentrations_mV']:.3f} mV")
    print(f"K only: {s3['K_only_mV']:.3f} mV; Na only: {s3['Na_only_mV']:.3f} mV")
    print(f"J_total(-70 mV), outward = {s3['J_outward_at_minus70_total_A_per_m2']:.6g} A/m^2")
    print(f"J_total(0 mV), outward = {s3['J_outward_at_zero_total_A_per_m2']:.6g} A/m^2")
    print(f"I_soma(-50 mV), inward = {s4['I_soma_inward_at_minus50_A'] * 1e12:.3f} pA")
    print(f"Differential G_soma(-50 mV) = {s4['slope_conductance_soma_at_minus50_S'] * 1e9:.3f} nS")
    print(f"Tau (1/e) = {s4['tau_one_over_e_ms']:.3f} ms; Fit = {s4['tau_exponential_fit_ms']:.3f} ms")
    print(f"40-channel demo spike at {s6['demonstration_stimulus_pA']:.3f} pA: {s6['fixed_seed_active_40_peak_mV']:.3f} mV")
    print(f"Monte-Carlo I50 = {s6['I50_pA']:.3f} pA ({s6['monte_carlo_trials_per_current']} trials per current)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and visualize all tasks from Biomodeling Lab 1."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lab1_results"),
        help="Directory for PNG, CSV, and JSON results.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=400,
        help="Monte Carlo trials per stimulus current for the spike probability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    summary = run_all(args.output_dir, args.trials)
    print_summary(summary)
    print(f"\nFigures and tables: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
