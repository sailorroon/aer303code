from __future__ import annotations

import argparse
import io
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import correlate


# ---------------------------------------------------------------------------
# Experimental constants and file locations
# ---------------------------------------------------------------------------

GAIN_PA_PER_V = 17838.0
ATMOSPHERIC_PRESSURE_PA = 101325.0  # Assumed standard atmosphere
GAMMA = 1.4
RELATIVE_PRESSURE_UNCERTAINTY = 0.01  # 1% of corrected absolute pressure

# Entries correspond to v0, v1, ..., v11.
PORT_LIST = [1, 2, 3, 5, 4, 6, 7, 8, 9, 11, 10, 12]
PORT_TO_CHANNEL = {
    physical_port: f"v{channel_index}"
    for channel_index, physical_port in enumerate(PORT_LIST)
}

PITOT_CHANNEL = PORT_TO_CHANNEL[12]

# Offset recordings.
BEFORE_NAME = "supersonic_Port07_offset_before_0.dat"
AFTER_NAME = "subsonic_Port01_offset_after_0.dat"

# Default data directory requested by the user.
DEFAULT_DATA_FOLDER = Path(
    r"C:\Users\roone\Downloads\Group07\Group07"
)

# All supplied geometry rows, including the two unnumbered upstream rows
# and the two downstream rows.
GEOMETRY_X_MM = np.array(
    [0.0, 11.6, 24.7, 36.6, 48.3, 61.0, 73.7, 86.4,
     99.1, 118.0, 209.5, 300.1],
    dtype=float,
)

GEOMETRY_AREA_RATIO = np.array(
    [1.58, 1.26, 1.06, 1.00, 1.05, 1.15, 1.23, 1.27,
     1.28, 1.30, 1.30, 1.30],
    dtype=float,
)

TAP_X_MM = {
    1: 24.7,
    2: 36.6,
    3: 48.3,
    4: 61.0,
    5: 73.7,
    6: 86.4,
    7: 99.1,
    8: 118.0,
    9: 209.5,
    10: 300.1,
}


# ---------------------------------------------------------------------------
# Reading the supplied .dat recordings
# ---------------------------------------------------------------------------

class RecordingSource:
    """Read .dat files from either a ZIP archive or an extracted folder."""

    def __init__(self, path: Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Data source does not exist: {self.path}"
            )

        self.zip_file = None

        if self.path.is_file():
            self.zip_file = ZipFile(self.path)
            self.files = {
                Path(name).name: name
                for name in self.zip_file.namelist()
                if name.lower().endswith(".dat")
            }
        else:
            self.files = {
                file_path.name: file_path
                for file_path in self.path.rglob("*.dat")
            }

    def read(self, filename: str) -> tuple[pd.DataFrame, datetime]:
        """Read one recording and return its data and start timestamp."""

        if filename not in self.files:
            raise FileNotFoundError(
                f"Could not find '{filename}' inside {self.path}"
            )

        if self.zip_file is not None:
            contents = self.zip_file.read(
                self.files[filename]
            ).decode("utf-8-sig")
        else:
            contents = Path(self.files[filename]).read_text(
                encoding="utf-8-sig"
            )

        lines = contents.splitlines()

        if len(lines) < 3:
            raise ValueError(f"Recording is empty or malformed: {filename}")

        start_time_text = lines[0].strip() + " " + lines[1].strip()
        start_time = datetime.strptime(
            start_time_text,
            "%d/%m/%Y %I:%M:%S %p",
        )

        header_row = None

        for row_index, line in enumerate(lines):
            columns = line.split()

            if columns and columns[0] == "Time" and "v11" in columns:
                header_row = row_index
                break

        if header_row is None:
            raise ValueError(
                f"Could not find the Time/v0...v11 header in {filename}"
            )

        data = pd.read_csv(
            io.StringIO(contents),
            sep=r"\s+",
            skiprows=header_row,
        )

        required_columns = ["Time"] + [
            f"v{index}" for index in range(12)
        ]

        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {filename}: {sorted(missing_columns)}"
            )

        data = data[required_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        if len(data) < 2:
            raise ValueError(
                f"Not enough valid data rows in {filename}"
            )

        return data, start_time

    def close(self):
        """Close the ZIP archive if one was opened."""

        if self.zip_file is not None:
            self.zip_file.close()


def recording_midpoint(
    start_time: datetime,
    data: pd.DataFrame,
) -> datetime:
    """Return the midpoint timestamp of a recording."""

    duration = float(data["Time"].iloc[-1])
    return start_time + timedelta(seconds=duration / 2)


# ---------------------------------------------------------------------------
# Sampling uncertainty and integral time scale
# ---------------------------------------------------------------------------

def mean_and_sampling_uncertainty(
    values,
    sample_interval: float = 1.0,
    max_samples: int | None = None,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Return:
        mean,
        standard error of the mean,
        effective independent sample count (N_eff),
        integral time scale / time constant (tau_s),
        sample standard deviation,
        lag-1 autocorrelation (rho_1),
        correlation factor (1 + 2 * sum(rho_k)).

    Positive-lag autocorrelation is summed until its first nonpositive value.
    The integral time scale is estimated as:
        tau = 0.5 * dt * (1 + 2 * sum_{k=1}^{K} rho_k)
    so that N_eff = N * dt / (2 * tau) = N / correlation_factor.
    N_eff is strictly capped so it cannot exceed the frequency-based
    number of samples (or raw valid sample count).
    """

    samples = np.asarray(values, dtype=float)
    samples = samples[np.isfinite(samples)]

    sample_count = len(samples)

    if sample_count < 2:
        raise ValueError("At least two valid readings are required.")

    upper_sample_bound = (
        float(min(sample_count, max_samples))
        if max_samples is not None
        else float(sample_count)
    )

    mean_value = float(np.mean(samples))
    sample_std = float(np.std(samples, ddof=1))

    if sample_std == 0:
        return (
            mean_value,
            0.0,
            upper_sample_bound,
            0.5 * sample_interval,
            0.0,
            0.0,
            1.0,
        )

    centered_samples = samples - mean_value

    autocovariance = correlate(
        centered_samples,
        centered_samples,
        mode="full",
        method="fft",
    )[sample_count - 1:]

    autocorrelation = autocovariance / autocovariance[0]
    lag1_autocorrelation = (
        float(autocorrelation[1]) if len(autocorrelation) > 1 else 0.0
    )

    nonpositive_lags = np.flatnonzero(autocorrelation[1:] <= 0)

    if len(nonpositive_lags) > 0:
        stop_index = int(nonpositive_lags[0] + 1)
    else:
        stop_index = sample_count

    positive_autocorr_sum = float(np.sum(autocorrelation[1:stop_index]))
    correlation_factor = max(1.0, 1.0 + 2.0 * positive_autocorr_sum)

    # Integral time constant tau_int = dt * (0.5 + sum(rho)) = 0.5 * dt * correlation_factor
    time_constant_s = max(
        0.5 * sample_interval,
        0.5 * sample_interval * correlation_factor,
    )

    # N_eff cannot exceed the frequency-based number of samples
    effective_count = upper_sample_bound / correlation_factor
    effective_count = min(
        upper_sample_bound,
        max(1.0, effective_count),
    )

    standard_error = sample_std / np.sqrt(effective_count) * 1.96

    return (
        mean_value,
        standard_error,
        effective_count,
        time_constant_s,
        sample_std,
        lag1_autocorrelation,
        correlation_factor,
    )


# ---------------------------------------------------------------------------
# Pressure calibration and uncertainty propagation
# ---------------------------------------------------------------------------

def calculate_absolute_pressure(
    measured_voltage: float,
    reference_voltage: float,
) -> float:
    """Convert a baseline-corrected voltage to absolute pressure."""

    voltage_difference = measured_voltage - reference_voltage

    return (
        ATMOSPHERIC_PRESSURE_PA
        + GAIN_PA_PER_V * voltage_difference
    )


def combine_pressure_uncertainties(
    pressure_pa: float,
    sampling_uncertainty_pa: float,
) -> tuple[float, float]:
    """
    Combine sampling uncertainty and the additional 1% pressure uncertainty.

    Returns:
        relative/calibration component,
        total pressure uncertainty.
    """

    relative_component = (
        RELATIVE_PRESSURE_UNCERTAINTY * abs(pressure_pa)
    )

    total_uncertainty = np.hypot(
        sampling_uncertainty_pa,
        relative_component,
    )

    return relative_component, total_uncertainty


def calibrate_channel(
    channel: str,
    data: pd.DataFrame,
    baseline: dict,
) -> dict:
    """
    Correct one channel using the average of the before and after
    atmospheric baseline recordings.

    Sampling uncertainty from the measurement and both baseline recordings
    is propagated into the pressure. The additional 1% pressure uncertainty
    is then combined independently.
    """

    sample_interval = float(data["Time"].diff().median())
    duration_s = float(data["Time"].iloc[-1]) - float(data["Time"].iloc[0])
    freq_based_samples = int(round(duration_s / sample_interval)) + 1
    max_allowed_samples = min(len(data), freq_based_samples)

    (
        measured_voltage,
        voltage_sem,
        effective_count,
        time_constant_s,
        sample_std,
        lag1_autocorr,
        correlation_factor,
    ) = mean_and_sampling_uncertainty(
        data[channel],
        sample_interval=sample_interval,
        max_samples=max_allowed_samples,
    )

    # Final verification guard: N_eff must never exceed frequency-based sample count
    effective_count = min(float(max_allowed_samples), effective_count)

    (
        baseline_before,
        uncertainty_before,
        baseline_after,
        uncertainty_after,
    ) = baseline[channel]

    # Use the arithmetic average of the before and after reference voltages.
    reference_voltage = 0.5 * (baseline_before + baseline_after)

    # Propagate the uncertainty of the average, assuming independent
    # before and after reference measurements.
    reference_uncertainty = 0.5 * np.sqrt(
        uncertainty_before**2 + uncertainty_after**2
    )

    pressure_pa = calculate_absolute_pressure(
        measured_voltage,
        reference_voltage,
    )

    # Propagate voltage sampling uncertainty through p = p_atm + G(V - Vref).
    pressure_sampling_uncertainty_pa = GAIN_PA_PER_V * np.sqrt(
        voltage_sem**2 + reference_uncertainty**2
    )

    relative_component_pa, total_pressure_uncertainty_pa = (
        combine_pressure_uncertainties(
            pressure_pa,
            pressure_sampling_uncertainty_pa,
        )
    )

    # These alternatives are retained as a reference-choice sensitivity check.
    pressure_before_only = calculate_absolute_pressure(
        measured_voltage,
        baseline_before,
    )

    pressure_after_only = calculate_absolute_pressure(
        measured_voltage,
        baseline_after,
    )

    return {
        "pressure_pa": pressure_pa,
        "sampling_uncertainty_pa": pressure_sampling_uncertainty_pa,
        "one_percent_uncertainty_pa": relative_component_pa,
        "total_uncertainty_pa": total_pressure_uncertainty_pa,
        "effective_count": effective_count,
        "freq_based_samples": max_allowed_samples,
        "time_constant_s": time_constant_s,
        "sample_std_voltage": sample_std,
        "lag1_autocorr": lag1_autocorr,
        "correlation_factor": correlation_factor,
        "pressure_before_only_pa": pressure_before_only,
        "pressure_after_only_pa": pressure_after_only,
    }


# ---------------------------------------------------------------------------
# Pitot pressure ratio and experimental Mach number
# ---------------------------------------------------------------------------

def rayleigh_pitot_ratio(mach: float) -> float:
    """
    Supersonic Rayleigh Pitot relation:
    post-normal-shock Pitot pressure divided by upstream static pressure.
    """

    gamma = GAMMA

    first_term = (
        (gamma + 1.0) * mach**2 / 2.0
    ) ** (gamma / (gamma - 1.0))

    second_term = (
        (2.0 * gamma * mach**2 - (gamma - 1.0))
        / (gamma + 1.0)
    ) ** (-1.0 / (gamma - 1.0))

    return first_term * second_term


def mach_from_pressure_ratio(
    ratio: float,
) -> tuple[float, str, float]:
    """
    Calculate Mach number from Pitot/static pressure ratio.

    Returns Mach number, branch description, and dM/dR.
    """

    if not np.isfinite(ratio) or ratio < 1.0:
        return np.nan, "INVALID: Pitot pressure below static", np.nan

    gamma = GAMMA

    critical_ratio = (
        (gamma + 1.0) / 2.0
    ) ** (gamma / (gamma - 1.0))

    if np.isclose(ratio, 1.0, rtol=0.0, atol=1e-12):
        return 0.0, "subsonic", np.nan

    if ratio <= critical_ratio:
        mach = np.sqrt(
            2.0 / (gamma - 1.0)
            * (ratio ** ((gamma - 1.0) / gamma) - 1.0)
        )

        derivative = (
            ratio ** (-1.0 / gamma)
            / (gamma * mach)
        )

        return float(mach), "subsonic isentropic", float(derivative)

    mach = brentq(
        lambda trial_mach: rayleigh_pitot_ratio(trial_mach) - ratio,
        1.0,
        10.0,
    )

    derivative_ratio = ratio * 2.0 * gamma / (gamma - 1.0) * (
        1.0 / mach
        - 2.0 * mach
        / (2.0 * gamma * mach**2 - (gamma - 1.0))
    )

    derivative = 1.0 / derivative_ratio

    return float(mach), "supersonic Rayleigh", float(derivative)


def calculate_experimental_mach(
    pitot_pressure_pa: float,
    static_pressure_pa: float,
    pitot_uncertainty_pa: float,
    static_uncertainty_pa: float,
) -> dict:
    """Calculate Mach, stagnation pressure, and propagate pressure uncertainties."""

    if pitot_pressure_pa <= 0 or static_pressure_pa <= 0:
        return {
            "mach": np.nan,
            "branch": "INVALID: nonpositive pressure",
            "ratio": np.nan,
            "ratio_uncertainty": np.nan,
            "mach_uncertainty": np.nan,
            "stagnation_pressure_pa": np.nan,
            "stagnation_uncertainty_pa": np.nan,
        }

    ratio = pitot_pressure_pa / static_pressure_pa

    mach, branch, derivative = mach_from_pressure_ratio(ratio)

    ratio_uncertainty = np.sqrt(
        (pitot_uncertainty_pa / static_pressure_pa) ** 2
        + (
            pitot_pressure_pa
            * static_uncertainty_pa
            / static_pressure_pa**2
        ) ** 2
    )

    if not np.isfinite(derivative):
        mach_uncertainty = np.nan
    else:
        mach_uncertainty = abs(derivative) * ratio_uncertainty

    # Local stagnation pressure p0 from static pressure and Mach (or Pitot in subsonic)
    if np.isfinite(mach):
        isen_factor = (
            1.0 + 0.5 * (GAMMA - 1.0) * mach**2
        ) ** (GAMMA / (GAMMA - 1.0))
        stagnation_pressure_pa = static_pressure_pa * isen_factor

        if branch.startswith("subsonic"):
            # In subsonic isentropic flow, p0 equals Pitot pressure directly
            stagnation_uncertainty_pa = pitot_uncertainty_pa
        else:
            # Propagate uncertainty through p0 = p_static * f(M)
            dp0_dp = isen_factor
            dp0_dM = (
                static_pressure_pa
                * GAMMA
                * mach
                * (1.0 + 0.5 * (GAMMA - 1.0) * mach**2) ** (1.0 / (GAMMA - 1.0))
            )
            stagnation_uncertainty_pa = float(
                np.hypot(
                    dp0_dp * static_uncertainty_pa,
                    dp0_dM * (mach_uncertainty if np.isfinite(mach_uncertainty) else 0.0),
                )
            )
    else:
        stagnation_pressure_pa = np.nan
        stagnation_uncertainty_pa = np.nan

    return {
        "mach": mach,
        "branch": branch,
        "ratio": ratio,
        "ratio_uncertainty": ratio_uncertainty,
        "mach_uncertainty": mach_uncertainty,
        "stagnation_pressure_pa": stagnation_pressure_pa,
        "stagnation_uncertainty_pa": stagnation_uncertainty_pa,
    }


# ---------------------------------------------------------------------------
# Isentropic theoretical distributions
# ---------------------------------------------------------------------------

def area_ratio_from_mach(mach: float) -> float:
    """Calculate A/A* from the isentropic area-Mach relation."""

    gamma = GAMMA

    term = (
        2.0 / (gamma + 1.0)
        * (1.0 + (gamma - 1.0) * mach**2 / 2.0)
    )

    exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0))

    return term**exponent / mach


def solve_area_mach_roots(
    area_ratio: float,
) -> tuple[float, float]:
    """
    Find both Mach roots for a supplied A/A*.

    The subsonic root lies between 0 and 1.
    The supersonic root lies above 1.
    """

    if area_ratio < 1.0:
        raise ValueError(
            f"A/A* must be at least 1. Received {area_ratio}."
        )

    if np.isclose(area_ratio, 1.0, atol=1e-12):
        return 1.0, 1.0

    equation = lambda mach: area_ratio_from_mach(mach) - area_ratio

    subsonic_root = brentq(
        equation,
        1e-6,
        1.0 - 1e-8,
    )

    supersonic_root = brentq(
        equation,
        1.0 + 1e-8,
        10.0,
    )

    return subsonic_root, supersonic_root


def isentropic_static_pressure(
    stagnation_pressure_pa: float,
    mach: float,
) -> float:
    """Calculate static pressure from p0/p and Mach."""

    pressure_factor = (
        1.0 + (GAMMA - 1.0) * mach**2 / 2.0
    ) ** (GAMMA / (GAMMA - 1.0))

    return stagnation_pressure_pa / pressure_factor


def calculate_theoretical_distributions(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate both theoretical Mach branches and their static pressures.

    One stagnation pressure is estimated for each regime from the first
    experimental station among stations 1-7:
        p0 = p_static * (1 + (gamma-1) M^2 / 2)^(gamma/(gamma-1))

    Mach calculations use datasets 1-7 and their corresponding static taps.
    Experimental static pressures at taps 8-10 are calculated exclusively
    from dataset Port01 for each flow regime. The theoretical pressure
    curves are evaluated at every supplied geometry position, including
    x = 118.0, 209.5, and 300.1 mm.
    The resulting p0 is held constant along that regime's isentropic curve.
    """

    theory_x_mm = GEOMETRY_X_MM
    theory_area_ratio = GEOMETRY_AREA_RATIO

    subsonic_mach = []
    supersonic_mach = []

    for area_ratio in theory_area_ratio:
        sub_root, super_root = solve_area_mach_roots(area_ratio)
        subsonic_mach.append(sub_root)
        supersonic_mach.append(super_root)

    # The supersonic branch is not physically selected upstream of the
    # throat. Keep the M=1 throat point, but omit the M>1 branch for x < 36.6 mm.
    supersonic_mach = np.asarray(supersonic_mach, dtype=float)
    supersonic_mach[theory_x_mm < 36.6] = np.nan

    theory = pd.DataFrame({
        "x_mm": theory_x_mm,
        "A_over_Astar": theory_area_ratio,
        "M_subsonic_theory": subsonic_mach,
        "M_supersonic_theory": supersonic_mach,
    })

    stagnation_pressures = {}

    for regime in ("subsonic", "supersonic"):
        first_station = (
            results[results["regime"] == regime]
            .sort_values("station")
            .iloc[0]
        )

        first_mach = float(first_station["Mach"])
        first_static_pressure = (
            float(first_station["p_static_kPa"]) * 1000.0
        )

        if not np.isfinite(first_mach):
            raise ValueError(
                f"Cannot estimate theoretical p0 for {regime}: "
                "the first measured Mach number is invalid."
            )

        stagnation_pressure = first_static_pressure * (
            1.0 + (GAMMA - 1.0) * first_mach**2 / 2.0
        ) ** (GAMMA / (GAMMA - 1.0))

        stagnation_pressures[regime] = stagnation_pressure

        theory[f"p_stagnation_{regime}_theory_kPa"] = (
            stagnation_pressure / 1000.0
        )

        theory_mach_values = theory[f"M_{regime}_theory"].to_numpy()

        theory[f"p_static_{regime}_theory_kPa"] = [
            (
                isentropic_static_pressure(stagnation_pressure, mach) / 1000.0
                if np.isfinite(mach)
                else np.nan
            )
            for mach in theory_mach_values
        ]

        theory[f"pressure_ratio_{regime}_theory"] = [
            (
                (
                    rayleigh_pitot_ratio(mach)
                    if mach > 1.0
                    else (1.0 + 0.5 * (GAMMA - 1.0) * mach**2) ** (GAMMA / (GAMMA - 1.0))
                )
                if np.isfinite(mach)
                else np.nan
            )
            for mach in theory_mach_values
        ]

    return theory, stagnation_pressures


def build_complete_output_table(
    results: pd.DataFrame,
    theory: pd.DataFrame,
    stagnation_pressures: dict,
) -> pd.DataFrame:
    """
    Merge experimental results and theoretical distributions into one complete
    summary DataFrame containing static pressure, stagnation pressure,
    pressure ratio, Mach number, theoretical pressure and Mach number,
    autocorrelation, and checked independent sample counts.
    """

    complete_rows = []
    theory_by_x = theory.set_index("x_mm")

    for _, row in results.iterrows():
        regime = row["regime"]
        x_val = float(row["x_mm"])

        if x_val in theory_by_x.index:
            th_row = theory_by_x.loc[x_val]
            area_ratio = float(th_row["A_over_Astar"])
            m_theory_regime = float(th_row[f"M_{regime}_theory"])
            p_theory_regime_kpa = float(th_row[f"p_static_{regime}_theory_kPa"])
            pr_theory_regime = float(th_row[f"pressure_ratio_{regime}_theory"])
            m_sub_theory = float(th_row["M_subsonic_theory"])
            m_sup_theory = float(th_row["M_supersonic_theory"])
            p_sub_theory_kpa = float(th_row["p_static_subsonic_theory_kPa"])
            p_sup_theory_kpa = float(th_row["p_static_supersonic_theory_kPa"])
        else:
            area_ratio = np.nan
            m_theory_regime = np.nan
            p_theory_regime_kpa = np.nan
            pr_theory_regime = np.nan
            m_sub_theory = np.nan
            m_sup_theory = np.nan
            p_sub_theory_kpa = np.nan
            p_sup_theory_kpa = np.nan

        freq_n = int(row["frequency_based_samples"])
        n_eff_static = min(float(freq_n), float(row["N_effective_static"]))
        n_eff_pitot = (
            min(float(freq_n), float(row["N_effective_pitot"]))
            if np.isfinite(row["N_effective_pitot"])
            else np.nan
        )

        # Verify N_eff <= N_freq
        n_eff_check_passed = bool(
            (n_eff_static <= freq_n)
            and (not np.isfinite(n_eff_pitot) or n_eff_pitot <= freq_n)
        )

        entry = row.to_dict()
        entry.update({
            "A_over_Astar": area_ratio,
            "p_stagnation_theory_reference_kPa": (
                stagnation_pressures[regime] / 1000.0
            ),
            "Mach_theory": m_theory_regime,
            "p_static_theory_kPa": p_theory_regime_kpa if "p_theory_regime_kpa" in locals() else p_theory_regime_kpa,
            "pressure_ratio_theory": pr_theory_regime,
            "M_subsonic_theory": m_sub_theory,
            "M_supersonic_theory": m_sup_theory,
            "p_static_subsonic_theory_kPa": p_sub_theory_kpa,
            "p_static_supersonic_theory_kPa": p_sup_theory_kpa,
            "N_effective_static": n_eff_static,
            "N_effective_pitot": n_eff_pitot,
            "N_effective_le_freq_samples_check": n_eff_check_passed,
        })
        complete_rows.append(entry)

    return pd.DataFrame(complete_rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AER303 pressure and Mach analysis"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help="Folder containing .dat files, or a ZIP archive.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory in which CSV files and plots will be saved.",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    source = RecordingSource(args.data)

    try:
        # ---------------------------------------------------------------
        # Read the two atmospheric baseline recordings
        # ---------------------------------------------------------------

        before_data, before_start = source.read(BEFORE_NAME)
        after_data, after_start = source.read(AFTER_NAME)

        before_dt = float(before_data["Time"].diff().median())
        after_dt = float(after_data["Time"].diff().median())

        before_duration = (
            float(before_data["Time"].iloc[-1])
            - float(before_data["Time"].iloc[0])
        )
        after_duration = (
            float(after_data["Time"].iloc[-1])
            - float(after_data["Time"].iloc[0])
        )

        before_freq_samples = min(
            len(before_data),
            int(round(before_duration / before_dt)) + 1,
        )
        after_freq_samples = min(
            len(after_data),
            int(round(after_duration / after_dt)) + 1,
        )

        before_midpoint = recording_midpoint(
            before_start,
            before_data,
        )

        after_midpoint = recording_midpoint(
            after_start,
            after_data,
        )

        if after_midpoint <= before_midpoint:
            raise ValueError(
                "The atmospheric baseline recordings are not chronological."
            )

        baseline = {}
        diagnostic_rows = []

        for channel_index in range(12):
            channel = f"v{channel_index}"

            (
                before_mean,
                before_sem,
                before_neff,
                before_tau,
                _,
                before_rho1,
                before_corr_factor,
            ) = mean_and_sampling_uncertainty(
                before_data[channel],
                sample_interval=before_dt,
                max_samples=before_freq_samples,
            )

            (
                after_mean,
                after_sem,
                after_neff,
                after_tau,
                _,
                after_rho1,
                after_corr_factor,
            ) = mean_and_sampling_uncertainty(
                after_data[channel],
                sample_interval=after_dt,
                max_samples=after_freq_samples,
            )

            baseline[channel] = (
                before_mean,
                before_sem,
                after_mean,
                after_sem,
            )

            diagnostic_rows.append({
                "channel": channel,
                "physical_port": PORT_LIST[channel_index],
                "baseline_before_V": before_mean,
                "baseline_after_V": after_mean,
                "baseline_average_V": 0.5 * (before_mean + after_mean),
                "drift_V": after_mean - before_mean,
                "equivalent_drift_kPa": (
                    GAIN_PA_PER_V * (after_mean - before_mean) / 1000.0
                ),
                "before_sampling_SEM_V": before_sem,
                "after_sampling_SEM_V": after_sem,
                "before_freq_based_samples": before_freq_samples,
                "before_effective_count": before_neff,
                "before_lag1_autocorr": before_rho1,
                "before_correlation_factor": before_corr_factor,
                "before_time_constant_s": before_tau,
                "after_freq_based_samples": after_freq_samples,
                "after_effective_count": after_neff,
                "after_lag1_autocorr": after_rho1,
                "after_correlation_factor": after_corr_factor,
                "after_time_constant_s": after_tau,
            })

        diagnostics = pd.DataFrame(diagnostic_rows)

        diagnostics_path = (
            args.output / "AER303_baseline_diagnostics.csv"
        )

        diagnostics.to_csv(diagnostics_path, index=False)

        print(f"Data source: {args.data}")
        print(
            f"Baseline reference centres: "
            f"{before_midpoint} to {after_midpoint}"
        )
        print(f"Sensor gain: {GAIN_PA_PER_V:.1f} Pa/V")
        print(
            f"Assumed atmospheric pressure: "
            f"{ATMOSPHERIC_PRESSURE_PA / 1000:.3f} kPa"
        )
        print(
            f"Additional independent pressure uncertainty: "
            f"{100 * RELATIVE_PRESSURE_UNCERTAINTY:.1f}%"
        )
        print(f"Static port mapping: {PORT_TO_CHANNEL}")
        print(f"Assumed traversing Pitot channel: {PITOT_CHANNEL}")
        print(
            "Static taps 8-10: pressure obtained exclusively from dataset "
            "Port01 separately for each flow regime."
        )

        print("\nBaseline drift, time constants, and independent samples:")
        print(
            diagnostics[
                [
                    "channel",
                    "physical_port",
                    "drift_V",
                    "equivalent_drift_kPa",
                    "before_time_constant_s",
                    "before_effective_count",
                    "after_time_constant_s",
                    "after_effective_count",
                ]
            ].round(4).to_string(index=False)
        )

        # ---------------------------------------------------------------
        # Process every experimental recording.
        # Datasets Port01-Port07 provide the Pitot/Mach measurements and
        # static-tap readings for taps 1-7.
        # Taps 8-10 are obtained solely from the Port01 dataset for both
        # supersonic and subsonic regimes.
        # ---------------------------------------------------------------

        result_rows = []
        # Port 8 pressure for each pitot-tube position (Port01 through Port07).
        port8_pitot_location_rows = []

        for regime in ("supersonic", "subsonic"):
            regime_recordings = {}

            for station in range(1, 8):
                filename = (
                    f"{regime}_Port{station:02d}_data_0.dat"
                )

                data, recording_start = source.read(filename)
                regime_recordings[station] = (data, filename)

                # Independently measure static pressure at physical port 8
                # for every dataset, where the dataset number identifies the
                # pitot-tube location (physical port 1 through 7).
                port8_result = calibrate_channel(
                    PORT_TO_CHANNEL[8],
                    data,
                    baseline,
                )
                port8_pitot_location_rows.append({
                    "regime": regime,
                    "pitot_tube_location": station,
                    "p_static_port8_kPa": port8_result["pressure_pa"] / 1000.0,
                    "u_p_static_port8_total_kPa": (
                        port8_result["total_uncertainty_pa"] / 1000.0
                    ),
                    "file": filename,
                })

                static_channel = PORT_TO_CHANNEL[station]

                static = calibrate_channel(
                    static_channel,
                    data,
                    baseline,
                )

                pitot = calibrate_channel(
                    PITOT_CHANNEL,
                    data,
                    baseline,
                )

                mach_result = calculate_experimental_mach(
                    pitot["pressure_pa"],
                    static["pressure_pa"],
                    pitot["total_uncertainty_pa"],
                    static["total_uncertainty_pa"],
                )

                warnings = []

                if not np.isfinite(mach_result["mach"]):
                    warnings.append("Invalid pressure ratio or pressure.")

                static_drift = (
                    baseline[static_channel][2]
                    - baseline[static_channel][0]
                )

                if abs(static_drift) > 0.05:
                    warnings.append(
                        "Large static-channel atmospheric baseline drift."
                    )

                if (
                    regime == "supersonic"
                    and mach_result["branch"].startswith("subsonic")
                ):
                    warnings.append(
                        "Nominally supersonic run gives subsonic local Mach."
                    )

                if (
                    static["effective_count"] < 30
                    or pitot["effective_count"] < 30
                ):
                    warnings.append(
                        "Small effective independent sample count."
                    )

                sample_interval = float(data["Time"].diff().median())
                duration_s = (
                    float(data["Time"].iloc[-1])
                    - float(data["Time"].iloc[0])
                )
                freq_based_samples = min(
                    len(data),
                    int(round(duration_s / sample_interval)) + 1,
                )

                result_rows.append({
                    "regime": regime,
                    "station": station,
                    "x_mm": TAP_X_MM[station],
                    "file": filename,
                    "static_channel": static_channel,
                    "pitot_channel": PITOT_CHANNEL,
                    "sample_rate_Hz": 1.0 / sample_interval,
                    "number_of_samples": len(data),
                    "frequency_based_samples": freq_based_samples,
                    "duration_s": duration_s,
                    "baseline_method": "average of before and after references",

                    "p_static_kPa": static["pressure_pa"] / 1000.0,
                    "u_p_static_sampling_kPa": (
                        static["sampling_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_static_1percent_kPa": (
                        static["one_percent_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_static_total_kPa": (
                        static["total_uncertainty_pa"] / 1000.0
                    ),

                    "p_pitot_kPa": pitot["pressure_pa"] / 1000.0,
                    "u_p_pitot_sampling_kPa": (
                        pitot["sampling_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_pitot_1percent_kPa": (
                        pitot["one_percent_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_pitot_total_kPa": (
                        pitot["total_uncertainty_pa"] / 1000.0
                    ),

                    "p_stagnation_kPa": (
                        mach_result["stagnation_pressure_pa"] / 1000.0
                    ),
                    "u_p_stagnation_total_kPa": (
                        mach_result["stagnation_uncertainty_pa"] / 1000.0
                    ),

                    "pressure_ratio": mach_result["ratio"],
                    "u_pressure_ratio": mach_result["ratio_uncertainty"],
                    "Mach": mach_result["mach"],
                    "flow_branch": mach_result["branch"],
                    "u_Mach_1.96sigma": mach_result["mach_uncertainty"],

                    "p_static_before_reference_kPa": (
                        static["pressure_before_only_pa"] / 1000.0
                    ),
                    "p_static_after_reference_kPa": (
                        static["pressure_after_only_pa"] / 1000.0
                    ),
                    "p_pitot_before_reference_kPa": (
                        pitot["pressure_before_only_pa"] / 1000.0
                    ),
                    "p_pitot_after_reference_kPa": (
                        pitot["pressure_after_only_pa"] / 1000.0
                    ),

                    "autocorr_lag1_static": static["lag1_autocorr"],
                    "autocorr_factor_static": static["correlation_factor"],
                    "tau_static_s": static["time_constant_s"],
                    "N_effective_static": static["effective_count"],
                    "autocorr_lag1_pitot": pitot["lag1_autocorr"],
                    "autocorr_factor_pitot": pitot["correlation_factor"],
                    "tau_pitot_s": pitot["time_constant_s"],
                    "N_effective_pitot": pitot["effective_count"],
                    "warnings": "; ".join(warnings),
                })

            # Taps 8, 9, and 10 are obtained solely from the Port01 dataset
            # for both supersonic and subsonic regimes.
            port01_data, port01_filename = regime_recordings[1]
            port01_sample_interval = float(
                port01_data["Time"].diff().median()
            )
            port01_duration_s = (
                float(port01_data["Time"].iloc[-1])
                - float(port01_data["Time"].iloc[0])
            )
            port01_freq_samples = min(
                len(port01_data),
                int(round(port01_duration_s / port01_sample_interval)) + 1,
            )

            for tap in range(8, 11):
                tap_channel = PORT_TO_CHANNEL[tap]
                tap_result = calibrate_channel(
                    tap_channel,
                    port01_data,
                    baseline,
                )

                warnings = [
                    "Static pressure obtained from Port01 dataset only"
                ]

                static_drift = (
                    baseline[tap_channel][2]
                    - baseline[tap_channel][0]
                )

                if abs(static_drift) > 0.05:
                    warnings.append(
                        "Large static-channel atmospheric baseline drift."
                    )

                if tap_result["effective_count"] < 30:
                    warnings.append(
                        "Small effective independent sample count."
                    )

                result_rows.append({
                    "regime": regime,
                    "station": tap,
                    "x_mm": TAP_X_MM[tap],
                    "file": port01_filename,
                    "static_channel": tap_channel,
                    "pitot_channel": PITOT_CHANNEL,
                    "sample_rate_Hz": 1.0 / port01_sample_interval,
                    "number_of_samples": len(port01_data),
                    "frequency_based_samples": port01_freq_samples,
                    "duration_s": port01_duration_s,
                    "baseline_method": "average of before and after references",
                    "p_static_kPa": tap_result["pressure_pa"] / 1000.0,
                    "u_p_static_sampling_kPa": (
                        tap_result["sampling_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_static_1percent_kPa": (
                        tap_result["one_percent_uncertainty_pa"] / 1000.0
                    ),
                    "u_p_static_total_kPa": (
                        tap_result["total_uncertainty_pa"] / 1000.0
                    ),
                    "p_pitot_kPa": np.nan,
                    "u_p_pitot_sampling_kPa": np.nan,
                    "u_p_pitot_1percent_kPa": np.nan,
                    "u_p_pitot_total_kPa": np.nan,
                    "p_stagnation_kPa": np.nan,
                    "u_p_stagnation_total_kPa": np.nan,
                    "pressure_ratio": np.nan,
                    "u_pressure_ratio": np.nan,
                    "Mach": np.nan,
                    "flow_branch": "not calculated for taps 8-10",
                    "u_Mach_1.96sigma": np.nan,
                    "p_static_before_reference_kPa": (
                        tap_result["pressure_before_only_pa"] / 1000.0
                    ),
                    "p_static_after_reference_kPa": (
                        tap_result["pressure_after_only_pa"] / 1000.0
                    ),
                    "p_pitot_before_reference_kPa": np.nan,
                    "p_pitot_after_reference_kPa": np.nan,
                    "autocorr_lag1_static": tap_result["lag1_autocorr"],
                    "autocorr_factor_static": tap_result["correlation_factor"],
                    "tau_static_s": tap_result["time_constant_s"],
                    "N_effective_static": tap_result["effective_count"],
                    "autocorr_lag1_pitot": np.nan,
                    "autocorr_factor_pitot": np.nan,
                    "tau_pitot_s": np.nan,
                    "N_effective_pitot": np.nan,
                    "warnings": "; ".join(warnings),
                })

        results = pd.DataFrame(result_rows)
        port8_pitot_location_results = pd.DataFrame(port8_pitot_location_rows)

        # Prepare the requested two-regime table for port 8, with each row
        # representing the physical port where the pitot tube was located.
        port8_table = (
            port8_pitot_location_results
            .pivot(index="pitot_tube_location", columns="regime",
                   values=["p_static_port8_kPa", "u_p_static_port8_total_kPa"])
            .sort_index()
        )
        port8_table.columns = [f"{value}_{regime}" for value, regime in port8_table.columns]
        port8_table = port8_table.rename(columns={
            "p_static_port8_kPa_subsonic": "P_subsonic_kPa",
            "u_p_static_port8_total_kPa_subsonic": "u_P_subsonic_kPa",
            "p_static_port8_kPa_supersonic": "P_supersonic_kPa",
            "u_p_static_port8_total_kPa_supersonic": "u_P_supersonic_kPa",
        }).reset_index()
        port8_table.insert(
            1,
            "Pitot tube location",
            port8_table["pitot_tube_location"].map(lambda port: f"Port {port}"),
        )
        port8_table = port8_table[[
            "Pitot tube location", "P_subsonic_kPa", "u_P_subsonic_kPa",
            "P_supersonic_kPa", "u_P_supersonic_kPa",
        ]]
        port8_table_path = args.output / "AER303_port8_pressure_by_pitot_location.csv"
        port8_table.to_csv(port8_table_path, index=False)

        results_path = args.output / "AER303_results_corrected.csv"
        results.to_csv(results_path, index=False)

        # ---------------------------------------------------------------
        # Calculate theoretical area-Mach and pressure distributions
        # ---------------------------------------------------------------

        theory, stagnation_pressures = (
            calculate_theoretical_distributions(results)
        )

        theory_path = args.output / "AER303_theoretical_distribution.csv"
        theory.to_csv(theory_path, index=False)

        # ---------------------------------------------------------------
        # Build and save the complete combined output CSV on top of
        # existing outputs
        # ---------------------------------------------------------------

        complete_output = build_complete_output_table(
            results,
            theory,
            stagnation_pressures,
        )

        complete_output_path = args.output / "AER303_complete_output.csv"
        complete_output.to_csv(complete_output_path, index=False)

        print("\nEstimated stagnation pressures used for theory:")
        for regime, pressure in stagnation_pressures.items():
            print(f"  {regime.title()}: {pressure / 1000:.3f} kPa")

        print("\nExperimental results (Pressures and Mach):")
        print(
            complete_output[
                [
                    "regime",
                    "station",
                    "x_mm",
                    "p_static_kPa",
                    "p_pitot_kPa",
                    "p_stagnation_kPa",
                    "pressure_ratio",
                    "Mach",
                    "p_static_theory_kPa",
                    "Mach_theory",
                ]
            ].round(4).to_string(index=False)
        )

        print("\nAutocorrelation, time constants, and independent sample counts:")
        print(
            complete_output[
                [
                    "regime",
                    "station",
                    "frequency_based_samples",
                    "autocorr_lag1_static",
                    "autocorr_factor_static",
                    "tau_static_s",
                    "N_effective_static",
                    "autocorr_lag1_pitot",
                    "N_effective_pitot",
                    "N_effective_le_freq_samples_check",
                ]
            ].round(4).to_string(index=False)
        )

        print(f"\nSaved experimental results: {results_path}")
        print(f"Saved baseline diagnostics: {diagnostics_path}")
        print(f"Saved theoretical distribution: {theory_path}")
        print(f"Saved complete combined output: {complete_output_path}")

        # ---------------------------------------------------------------
        # Generate plots with experimental vertical error bars
        # ---------------------------------------------------------------

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "\nMatplotlib is not installed. Install it using "
                "'pip install matplotlib' to generate plots."
            )
            return

        regime_styles = {
            "subsonic": "o",
            "supersonic": "s",
        }

        # Mach number plot.
        fig, axis = plt.subplots(figsize=(9, 5.5))

        for regime in ("subsonic", "supersonic"):
            experimental = (
                results[results["regime"] == regime]
                .sort_values("x_mm")
            )

            axis.errorbar(
                experimental["x_mm"],
                experimental["Mach"],
                yerr=experimental["u_Mach_1.96sigma"],
                marker=regime_styles[regime],
                linestyle="-",
                capsize=4,
                label=f"{regime.title()} experiment",
            )

        mach_theory = theory[theory["x_mm"] <= 99.1]

        axis.plot(
            mach_theory["x_mm"],
            mach_theory["M_subsonic_theory"],
            linestyle="--",
            marker="o",
            label="Subsonic isentropic theory",
        )

        axis.plot(
            mach_theory["x_mm"],
            mach_theory["M_supersonic_theory"],
            linestyle="--",
            marker="s",
            label="Supersonic isentropic theory",
        )

        axis.axvline(
            36.6,
            linestyle=":",
            linewidth=1.2,
            label="Geometric throat",
        )

        axis.set_xlabel("Axial position, x (mm)")
        axis.set_ylabel("Mach number, M")
        axis.set_title("Experimental and Theoretical Mach Distribution")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        mach_plot_path = args.output / "AER303_Mach_distribution.png"
        fig.savefig(mach_plot_path, dpi=200)
        plt.close(fig)

        # Static pressure plot.
        fig, axis = plt.subplots(figsize=(9, 5.5))

        for regime in ("subsonic", "supersonic"):
            experimental = (
                results[results["regime"] == regime]
                .sort_values("x_mm")
            )

            axis.errorbar(
                experimental["x_mm"],
                experimental["p_static_kPa"],
                yerr=experimental["u_p_static_total_kPa"],
                marker=regime_styles[regime],
                linestyle="-",
                capsize=4,
                label=f"{regime.title()} experiment (taps 1-10)",
            )

            theoretical_pressure_column = (
                f"p_static_{regime}_theory_kPa"
            )

            axis.plot(
                theory["x_mm"],
                theory[theoretical_pressure_column],
                linestyle="--",
                marker="o" if regime == "subsonic" else "s",
                label=f"{regime.title()} isentropic theory",
            )

        axis.axvline(
            36.6,
            linestyle=":",
            linewidth=1.2,
            label="Geometric throat",
        )

        axis.set_xlabel("Axial position, x (mm)")
        axis.set_ylabel("Static pressure, p (kPa)")
        axis.set_title("Experimental and Theoretical Static Pressure")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()

        pressure_plot_path = (
            args.output / "AER303_static_pressure_distribution.png"
        )

        fig.savefig(pressure_plot_path, dpi=200)
        plt.close(fig)

        # Requested port 8 pressure comparison: plot only the seven individual
        # points per regime, with total pressure uncertainty as vertical bars.
        fig, axis = plt.subplots(figsize=(9, 5.5))
        plot_styles = {
            "subsonic": {"marker": "o", "color": "tab:blue"},
            "supersonic": {"marker": "s", "color": "tab:orange"},
        }
        for regime in ("subsonic", "supersonic"):
            subset = port8_pitot_location_results[
                port8_pitot_location_results["regime"] == regime
            ].sort_values("pitot_tube_location")
            style = plot_styles[regime]
            axis.errorbar(
                subset["pitot_tube_location"],
                subset["p_static_port8_kPa"],
                yerr=subset["u_p_static_port8_total_kPa"],
                marker=style["marker"],
                color=style["color"],
                linestyle="none",
                capsize=4,
                label=f"{regime.title()} pressure at port 8",
            )

        axis.set_xticks(range(1, 8), [f"Port {port}" for port in range(1, 8)])
        axis.set_xlabel("Pitot tube location")
        axis.set_ylabel("Static pressure at port 8, p (kPa)")
        axis.set_title("Port 8 Static Pressure vs. Pitot Tube Location")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        port8_plot_path = args.output / "AER303_port8_pressure_by_pitot_location.png"
        fig.savefig(port8_plot_path, dpi=200)
        plt.close(fig)

        print("\nPort 8 static pressure by pitot tube location:")
        print(port8_table.round(4).to_string(index=False))
        print(f"Saved port 8 pressure table: {port8_table_path}")
        print(f"Saved port 8 pressure plot: {port8_plot_path}")

        print(f"Saved Mach plot: {mach_plot_path}")
        print(f"Saved pressure plot: {pressure_plot_path}")

        print("\nUncertainty summary:")
        print("  Pressure sampling uncertainty uses N_effective and tau.")
        print("  Independent 1% pressure uncertainty is included.")
        print("  Mach uncertainty propagates both total pressure uncertainties.")
        print("  Error bars show 1-sigma propagated experimental uncertainty.")
        print(
            "\nCaution: the theoretical curves assume isentropic flow and "
            "constant p0. They do not model stagnation-pressure loss across "
            "a shock. Baseline interpolation, atmospheric pressure, and "
            "calibration covariance are also not fully quantified."
        )

    finally:
        source.close()


if __name__ == "__main__":
    main()