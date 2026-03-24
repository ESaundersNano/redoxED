import pandas as pd
import numpy as np
import copy


class CyclingData:
    """ """

    def __init__(
        self,
        cycle_data: pd.DataFrame | None = None,
        time_series: pd.DataFrame | None = None,
        label: str | None = None,
    ) -> None:
        """ """
        self.label = label

        self.area: float | None = None  # electrode geometric area in m2 if known

        # Store the data if given
        self._cycle_summary = (
            cycle_data.copy() if cycle_data is not None else pd.DataFrame()
        )
        self._full_data = time_series.copy() if time_series is not None else None

        # Validate that data is consistent
        self._validate()

    @classmethod
    def from_V_t_dq_cycle(
        cls,
        V_cell: np.ndarray,
        time: np.ndarray,
        dq: np.ndarray,
        cycle_number: np.ndarray,
        label: str | None = None,
        reverse_polarity: bool = False,
    ) -> "CyclingData":
        """
        Create CyclingData from raw electrochemical arrays.
        """

        def _classify_dq(dq_value):
            if dq_value > 0:
                return "positive"  # usually charge
            elif dq_value < 0:
                return "negative"  # usually discharge
            elif dq_value == 0:
                return "ocv"
            else:
                return "unknown"

        # Classify the current mode for each data point
        I_mode = np.array([_classify_dq(dq_val) for dq_val in dq])

        # Check for unknown values and raise error
        if np.any(I_mode == "unknown"):
            unknown_indices = np.where(I_mode == "unknown")[0]
            raise ValueError(f"Unknown I mode found at indices: {unknown_indices}.")

        # Calculate current (I_cell = dq/dt)
        dt = np.diff(time)
        dt = np.insert(dt, 0, dt[0] if len(dt) > 0 else 1)  # Handle first point
        # Handle zero time differences to avoid division by zero
        dt_safe = np.where(
            dt == 0, np.finfo(float).eps, dt
        )  # Replace 0 with tiny number
        I_cell = dq / dt_safe

        # Calculate incremental energy (dE = V * dq)
        dE = V_cell * dq

        # Calculate voltage integrand
        Vdt = V_cell * dt

        # Create DataFrame for easier manipulation
        df = pd.DataFrame(
            {
                "time": time,
                "V_cell": V_cell,
                "dq": dq,
                "cycle_number": cycle_number,
                "I_mode": I_mode,
                "dt": dt,
                "I_cell": I_cell,
                "dE": dE,
                "Vdt": Vdt,
            }
        )

        # Initialise time series cumulative columns
        df["Q_pos"] = 0.0
        df["Q_neg"] = 0.0
        df["E_pos"] = 0.0  # Cumulative energy positive
        df["E_neg"] = 0.0  # Cumulative energy negative

        # Initialize lists to store averages in the same order as cycles
        V_cell_avg_pos_list = []
        V_cell_avg_neg_list = []
        I_cell_avg_pos_list = []
        I_cell_avg_neg_list = []

        for cycle_num in df["cycle_number"].unique():
            # Get mask for this cycle
            cycle_mask = df["cycle_number"] == cycle_num
            cycle_data = df.loc[cycle_mask]

            # Create masks for different modes within this cycle so OCV doesn't affect sums
            pos_mask = cycle_data["I_mode"] == "positive"
            neg_mask = cycle_data["I_mode"] == "negative"

            # Calculate cumulative sums for this cycle only
            Q_pos = (cycle_data["dq"] * pos_mask).cumsum()
            Q_neg = (cycle_data["dq"] * neg_mask).cumsum()
            E_pos = (cycle_data["dE"] * pos_mask).cumsum()
            E_neg = (cycle_data["dE"] * neg_mask).cumsum()

            # Average voltage and current during charge/discharge
            Vdt_sum_pos = (cycle_data["Vdt"] * pos_mask).cumsum()
            Vdt_sum_neg = (cycle_data["Vdt"] * neg_mask).cumsum()
            t_pos = (cycle_data["dt"] * pos_mask).cumsum()
            t_neg = (cycle_data["dt"] * neg_mask).cumsum()
            V_cell_avg_pos = (
                Vdt_sum_pos.iloc[-1] / t_pos.iloc[-1] if t_pos.iloc[-1] > 0 else np.nan
            )
            V_cell_avg_neg = (
                Vdt_sum_neg.iloc[-1] / t_neg.iloc[-1] if t_neg.iloc[-1] > 0 else np.nan
            )
            I_cell_avg_pos = (
                Q_pos.iloc[-1] / t_pos.iloc[-1] if t_pos.iloc[-1] > 0 else np.nan
            )
            I_cell_avg_neg = (
                Q_neg.iloc[-1] / t_neg.iloc[-1] if t_neg.iloc[-1] > 0 else np.nan
            )
            # Append averages to lists
            V_cell_avg_pos_list.append(V_cell_avg_pos)
            V_cell_avg_neg_list.append(V_cell_avg_neg)
            I_cell_avg_pos_list.append(I_cell_avg_pos)
            I_cell_avg_neg_list.append(I_cell_avg_neg)

            # Assign back to the main dataframe
            df.loc[cycle_mask, "Q_pos"] = Q_pos
            df.loc[cycle_mask, "Q_neg"] = Q_neg
            df.loc[cycle_mask, "E_pos"] = E_pos
            df.loc[cycle_mask, "E_neg"] = E_neg

        # Create cycle summary using groupby aggregation
        cycle_summary = (
            df.groupby("cycle_number")
            .agg(
                {
                    "Q_pos": "last",  # Final value is the total
                    "Q_neg": "last",
                    "E_pos": "last",
                    "E_neg": "last",
                }
            )
            .round(6)
        )

        # Flatten column names and reset index
        cycle_summary.columns = [
            "Q_pos_total",
            "Q_neg_total",
            "E_pos_total",
            "E_neg_total",
        ]
        cycle_summary = cycle_summary.reset_index()
        # cycle_summary.rename(columns={'cycle_number': 'cycle'}, inplace=True)

        if reverse_polarity == False:
            # Average voltage and current during charge/discharge
            cycle_summary["V_cell_avg_charge"] = V_cell_avg_pos_list
            cycle_summary["V_cell_avg_discharge"] = V_cell_avg_neg_list
            cycle_summary["I_cell_avg_charge"] = I_cell_avg_pos_list
            cycle_summary["I_cell_avg_discharge"] = I_cell_avg_neg_list
            # Add convenience columns
            cycle_summary["Q_charge"] = cycle_summary["Q_pos_total"]
            cycle_summary["Q_discharge"] = abs(cycle_summary["Q_neg_total"])
            cycle_summary["E_charge"] = cycle_summary["E_pos_total"]
            cycle_summary["E_discharge"] = abs(cycle_summary["E_neg_total"])

        elif reverse_polarity == True:
            # Average voltage and current during charge/discharge
            cycle_summary["V_cell_avg_charge"] = V_cell_avg_neg_list
            cycle_summary["V_cell_avg_discharge"] = V_cell_avg_pos_list
            cycle_summary["I_cell_avg_charge"] = I_cell_avg_neg_list
            cycle_summary["I_cell_avg_discharge"] = I_cell_avg_pos_list
            # Add convenience columns
            cycle_summary["Q_charge"] = abs(cycle_summary["Q_neg_total"])
            cycle_summary["Q_discharge"] = cycle_summary["Q_pos_total"]
            cycle_summary["E_charge"] = abs(cycle_summary["E_neg_total"])
            cycle_summary["E_discharge"] = cycle_summary["E_pos_total"]

        # Calculate Coulombic Efficiency
        cycle_summary["CE"] = (
            abs(cycle_summary["Q_discharge"] / cycle_summary["Q_charge"]) * 100
        )
        # Calculate Voltage Efficiency
        cycle_summary["VE"] = (
            abs(
                cycle_summary["V_cell_avg_discharge"]
                / cycle_summary["V_cell_avg_charge"]
            )
            * 100
        )
        # Calculate Energy Efficiency
        cycle_summary["EE"] = (
            abs(cycle_summary["E_discharge"] / cycle_summary["E_charge"]) * 100
        )

        return cls(cycle_data=cycle_summary, time_series=df, label=label)

    def set_area(self, area: float = 0.0005) -> None:
        """
        Set the electrode geometric area for current density calculations.

        Parameters:
            area: Electrode area in m²
        """
        if area <= 0:
            raise ValueError("Area must be a positive value.")
        self.area = area

    def add_computed_metrics(self, **metrics):
        """
        Add additional computed metrics to existing cycle data.

        Parameters:
            **metrics: Keyword arguments for new metric arrays

        Example:
            cycling_data.add_computed_metrics(
                power_fade=power_fade_array,
                temperature=temp_array
            )
        """
        for metric_name, metric_values in metrics.items():
            if len(metric_values) != len(self._cycle_summary):
                raise ValueError(
                    f"Metric '{metric_name}' length ({len(metric_values)}) "
                    f"doesn't match cycle data length ({len(self._cycle_summary)})"
                )
            self._cycle_summary[metric_name] = metric_values

    def get_cycle(self, cycle_num: int) -> pd.DataFrame:
        """
        Get all time-series data for a specific cycle.

        Parameters:
            cycle_num: The cycle number to extract

        Returns:
            DataFrame containing all time points for that cycle
        """
        if self._full_data is None:
            raise ValueError(
                "No time-series data available. This dataset only contains cycle summaries."
            )

        cycle_data = self._full_data[
            self._full_data[self.cycle_column] == cycle_num
        ].copy()

        if len(cycle_data) == 0:
            raise ValueError(f"No data found for cycle {cycle_num}")

        return cycle_data

    def get_cycle_summary(self, cycle_num: int) -> pd.Series:
        """
        Get summary metrics for a specific cycle.

        Parameters:
            cycle_num: The cycle number to get summary for

        Returns:
            Series containing CE, VE, capacity, etc. for that cycle
        """
        if self._cycle_summary.empty:
            raise ValueError("No cycle summary data available")

        cycle_rows = self._cycle_summary[self._cycle_summary["cycle"] == cycle_num]
        if len(cycle_rows) == 0:
            raise ValueError(f"No summary data found for cycle {cycle_num}")

        return cycle_rows.iloc[0]

    def _validate(self) -> None:
        """Validate the cycling data for consistency."""
        # Check cycle summary data
        if not self._cycle_summary.empty:
            if "cycle_number" not in self._cycle_summary.columns:
                raise ValueError(
                    "Cycle summary data must contain 'cycle_number' column"
                )
        pass

    # Properties for easy access to cycle-level data (Level 1)
    @property
    def cycle_data(self) -> pd.DataFrame:
        """Per-cycle summary data - use this for plotting efficiency trends"""
        return self._cycle_summary

    @property
    def time_series(self) -> pd.DataFrame | None:
        """Full time-series data - use this for detailed within-cycle analysis"""
        return self._full_data

    @property
    def cycles(self) -> np.ndarray:
        """Array of available cycle numbers"""
        if not self._cycle_summary.empty:
            return self._cycle_summary["cycle"].values
        elif self._full_data is not None:
            return self._full_data[self.cycle_column].unique()
        else:
            return np.array([])

    # Properties for common efficiency metrics - these now just access stored data
    @property
    def CE(self) -> np.ndarray | None:
        """Coulombic efficiency by cycle - from pre-computed data"""
        return (
            self._cycle_summary["CE"].values
            if "CE" in self._cycle_summary.columns
            else None
        )

    def __repr__(self) -> str:
        """String representation of the CyclingData object"""
        n_cycles = len(self.cycles)
        max_cycles = np.max(self.cycles)
        has_timeseries = self._full_data is not None
        label_str = f", label='{self.label}'" if self.label else ""

        return f"CyclingData({n_cycles} cycles of {max_cycles}, time_series={has_timeseries}{label_str})"

    def copy(self):
        """Return a deep copy of this DRTData instance."""
        return copy.deepcopy(self)
