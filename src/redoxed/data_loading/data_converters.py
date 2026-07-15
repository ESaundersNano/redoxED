import numpy as np
import pandas as pd

from redoxed.dc import CyclingData, PolarisationData
from redoxed.impedance import EISData

"""Helpers for converting tabular EC data into redoxed domain objects."""


def df_to_EISData(
    df: pd.DataFrame,
    label: str | None = None,
    column_mapping: dict[str, str | None] | None = None,
) -> EISData:
    """Convert an EC-Lab style DataFrame into an EISData object.
    Assumes that the DataFrame contains columns for frequency and impedance, with optional column names
    for real and imaginary parts. Column names can be customized using the `column_mapping` parameter.
    If no mapping is provided, default EC-Lab column names are used.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing impedance data.
    label : str, optional
        Optional label for the returned object.
    column_mapping : dict[str, str | None], optional
        Mapping from semantic names to DataFrame column names.

    Returns
    -------
    EISData
        An EISData instance created from the input DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or the data cannot be parsed.
    NotImplementedError
        If the provided column mapping does not describe a supported impedance layout.
    """
    # Default to the standard EC-Lab export column names.
    default_mapping: dict[str, str | None] = {
        "Z_re": "Re(Z)/Ohm",
        "-Z_im": "-Im(Z)/Ohm",
        "f": "freq/Hz",
        "Z": None,
        "Z_im": None,
    }
    if column_mapping is None:
        column_mapping = default_mapping

    # Keep only the columns that are actually required for the selected layout.
    required_columns: dict[str, str] = {
        key: value for key, value in column_mapping.items() if value is not None
    }
    if not all(col in df.columns for col in list(required_columns.values())):
        raise ValueError(f"DataFrame did not contain the columns: {required_columns}")
    elif all(key in required_columns for key in ["Z_re", "-Z_im", "f"]):
        Z = (
            df[required_columns["Z_re"]].to_numpy()
            - 1j * df[required_columns["-Z_im"]].to_numpy()
        )
    elif all(key in required_columns for key in ["Z", "f"]):
        Z = df[required_columns["Z"]].to_numpy()
    elif all(key in required_columns for key in ["Z_re", "Z_im", "f"]):
        Z = (
            df[required_columns["Z_re"]].to_numpy()
            + 1j * df[required_columns["Z_im"]].to_numpy()
        )
    else:
        raise NotImplementedError(
            "Unsupported data format. Try extract necessary data from ECData's df."
        )
    f = df[required_columns["f"]].to_numpy()

    return EISData(Z, f, label=label)


# unlike df_to_EISData, this function is specific to Biologic GCPL data format/default units and not intended to be flexible to other formats. Users likely have to write their own for other file formats.
def df_to_CyclingData(
    df: pd.DataFrame,
    label: str | None = None,
) -> CyclingData:
    """Convert a Biologic GCPL DataFrame into a CyclingData object.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with Biologic GCPL cycling columns.
    label : str, optional
        Optional label for the returned object.

    Returns
    -------
    CyclingData
        A CyclingData instance built from the raw DataFrame.
    """
    # Extract the raw quantities used by the domain object.
    V_cell = df["Ewe/V"].to_numpy()
    time = df["time/s"].to_numpy()
    dq = df["dq/mA.h"].to_numpy() * 3.6  # convert to Coulombs

    # Biologic half-cycles are 0-indexed in exports; shift them to 1-indexed cycles.
    cycle_number = df["half cycle"] // 2
    cycle_number = cycle_number + 1

    # Ns (sequence number) does not describe current within a step.
    # control/V/mA can drift out of sync near step boundaries, but dq/mA.h still
    # tracks the signed charge transfer needed for the current calculations.
    # Q charge/discharge/mA.h is cumulative within a half-cycle.
    # (Q-Qo)/mA.h is cumulative over the whole run and tracks charge loss.

    return CyclingData.from_V_t_dq_cycle(
        V_cell=V_cell, time=time, dq=dq, cycle_number=cycle_number, label=label
    )


# unlike df_to_EISData, this function is specific to Biologic GCPL data format/default units and not intended to be flexible to other formats. Users likely have to write their own for other file formats.
def df_to_PolarisationData(
    df: pd.DataFrame,
    A: float = 5,
    pulse_index_range: tuple[int | None, int | None] = (-5, None),
    label: str | None = None,
) -> PolarisationData:
    """Convert pulse data into a PolarisationData object.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing Biologic polarisation data.
    A : float, optional
        Electrode area in cm^2.
    pulse_index_range : tuple[int | None, int | None], optional
        Slice bounds used to select the part of each pulse to average.
    label : str, optional
        Optional label for the returned object.

    Returns
    -------
    PolarisationData
        A PolarisationData instance built from the input pulses.
    """
    V: list[float] = []
    j: list[float] = []
    pulse_range = slice(pulse_index_range[0], pulse_index_range[1])
    # Group by half cycle so each pulse contributes one averaged point.
    for _, group in df.groupby("half cycle"):

        # The last row often contains the next step's control value, so drop it.
        group = group.iloc[:-1]

        # If all control values are zero, keep the full group.
        # Otherwise remove the zero rows so we average only the active pulse segment.
        if group["control/V/mA"].eq(0).all():
            selected_group = group
        else:
            selected_group = group[group["control/V/mA"] != 0]

        # Apply the same slice to every pulse so the averaging window is consistent.
        selected_group = selected_group.iloc[pulse_range]

        # Average the selected window into one voltage/current-density point.
        avg_Ewe = selected_group["Ewe/V"].mean()
        avg_control_I = selected_group["control/V/mA"].mean()
        avg_control_j = avg_control_I / A

        V.append(avg_Ewe)
        j.append(avg_control_j)
    V = np.array(V)
    j = np.array(j)

    return PolarisationData(V, j, A, label=label)


# unlike df_to_EISData, this function is specific to Biologic GCPL data format/default units and not intended to be flexible to other formats. Users likely have to write their own for other file formats.
def df_to_PolarisationData2(
    df: pd.DataFrame,
    A: float = 5,
    pulse_index_range: tuple[int | None, int | None] = (-5, None),
    label: str | None = None,
) -> PolarisationData:
    """Convert pulse data by grouping on control current instead of half cycle. Useful when the half cycle column is not available or when the user wants to group by current steps instead of half cycles.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing Biologic polarisation data.
    A : float, optional
        Electrode area in cm^2.
    pulse_index_range : tuple[int | None, int | None], optional
        Slice bounds used to select the part of each pulse to average.
    label : str, optional
        Optional label for the returned object.

    Returns
    -------
    PolarisationData
        A PolarisationData instance built from the grouped pulses.
    """
    V: list[float] = []
    j: list[float] = []
    pulse_range = slice(pulse_index_range[0], pulse_index_range[1])

    # Group by control current values to identify distinct current steps.
    for control_current, group in df.groupby("control/V/mA"):

        # Skip groups that are too short for the requested slice.
        # This avoids averaging windows that would otherwise be empty or partial.
        if (
            len(group) < abs(pulse_index_range[0])
            if pulse_index_range[0] is not None
            else 1
        ):
            continue

        # Reuse the same slice window used in the half-cycle-based helper.
        selected_group = group.iloc[pulse_range]

        # Average the selected window into one voltage/current-density point.
        avg_Ewe = selected_group["Ewe/V"].mean()
        avg_control_I = selected_group["control/V/mA"].mean()
        avg_control_j = avg_control_I / A

        V.append(avg_Ewe)
        j.append(avg_control_j)

    V = np.array(V)
    j = np.array(j)

    return PolarisationData(V, j, A, label=label)
