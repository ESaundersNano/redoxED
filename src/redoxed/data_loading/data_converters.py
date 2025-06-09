from redoxed.impedance import EISData
from redoxed.DC import PolarisationData
import pandas as pd
import numpy as np

# a collection of functions to convert data to different formats by the preference of the author.


def df_to_EISData(
    df: pd.DataFrame,
    label: str | None = None,
    column_mapping: dict[str, str | None] | None = None,
) -> EISData:
    """ """
    # Set default column names (EClab file assumed)
    default_mapping = {
        "Z_re": "Re(Z)/Ohm",
        "-Z_im": "-Im(Z)/Ohm",
        "f": "freq/Hz",
        "Z": None,
        "Z_im": None,
    }
    if column_mapping is None:
        column_mapping = default_mapping

    # Check if the DataFrame contains the required columns
    # Drop entries with value None
    required_columns = {
        key: value for key, value in column_mapping.items() if value is not None
    }
    # Perform check for required columns
    if not all(col in df.columns for col in list(required_columns.values())):
        raise ValueError(f"DataFrame did not contain the columns: {required_columns}")
    # Convert the DataFrame columns to extract Z and f
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

    # Create and return an instance of EISData
    return EISData(Z, f, label=label)


def df_to_PolarisationData(
    df: pd.DataFrame,
    A: float,
    pulse_index_range: tuple = (-5, None),
    label: str | None = None,
) -> PolarisationData:
    """
    make sure to note units
    At the moment assumes provide indices of which parts of pulse you want to keep, applying the same to all.
    have to make sure also have wisely selected which bits of pulse are keeping.
    Am for now trusting the control I values even though not what is actually recorded.
    DO I need to sort my data so that puts negative and positive pulses in order?
    """
    V = []
    j = []
    pulse_range = slice(pulse_index_range[0], pulse_index_range[1])
    # Group by 'half cycle'
    for half_cycle, group in df.groupby("half cycle"):

        # account for value of control for next half cycle being at end of half cycle by removing it
        group = group.iloc[:-1]

        # Check if all 'control/V/mA' values are 0
        if group["control/V/mA"].eq(0).all():
            # If all values are 0, don't filter, use the whole group
            selected_group = group
        else:
            # Otherwise, filter for nonzero 'control/V/mA'
            selected_group = group[group["control/V/mA"] != 0]

        # Select the range of points you want to analyse
        selected_group = selected_group.iloc[pulse_range]  # apply slice
        # print(selected_group)

        # Compute averages for 'Ewe/V' and 'control/V/mA'
        avg_Ewe = selected_group["Ewe/V"].mean()
        avg_control_I = selected_group["control/V/mA"].mean()
        avg_control_j = avg_control_I / A

        V.append(avg_Ewe)
        j.append(avg_control_j)
    V = np.array(V)
    j = np.array(j)

    return PolarisationData(V, j, A, label=label)
