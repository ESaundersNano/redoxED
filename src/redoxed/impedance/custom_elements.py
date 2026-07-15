"""Custom electrochemical circuit elements and impedance functions.

Provides user-defined circuit elements for pyimpspec (e.g., SomeNewElement) and
custom impedance models including Transmission Line Model (TLM) implementations
for porous electrodes with various boundary conditions.
"""

from numpy import pi, inf
from numpy import float64, complex128
from numpy.typing import NDArray

import pyimpspec

from pyimpspec import (
    ComplexImpedances,  # Alias for a NumPy array of complex128 values
    Frequencies,  # Alias for a NumPy array of float64 values
    Circuit,
    Element,  # The base class for all circuit elements
    ElementDefinition,  # A class that contains information regarding a new circuit element
    ParameterDefinition,  # A class that contains information regarding a circuit element's parameter
    register_element,  # A function that processes the new element class
    parse_cdc,
)

pyimpspec.circuit.registry.reset(
    elements=True, default_parameters=True
)  # reset registry of elements and parameters


class SomeNewElement(Element):
    """User-defined circuit element implementing a power law impedance.

    Attributes:
        X (float): Impedance magnitude parameter with custom units.
        a (float): Exponent parameter controlling frequency dependence (0 < a \u2264 1).
    """

    def _impedance(self, f: Frequencies, X: float, a: float) -> ComplexImpedances:
        """Calculate impedance as a function of frequency and parameters.

        Parameters:
            f (Frequencies): Frequencies in Hz (Hertz).
            X (float): Impedance magnitude in custom units.
            a (float): Exponent for frequency dependence (dimensionless).

        Returns:
            ComplexImpedances: Complex impedance Z(f) in custom units.
        """
        return 1 / (X * (1j * 2 * pi * f) ** a)


register_element(
    ElementDefinition(
        Class=SomeNewElement,
        symbol="Ude",
        name="Some new element",
        description="User-defined element of some type.",
        equation="1/(X*(2*pi*f*I)^a)",
        parameters=[
            ParameterDefinition(
                symbol="X",
                unit="Some unit",
                description="Description of this parameter",
                value=1e-6,
                lower_limit=1e-24,
                upper_limit=inf,
                fixed=False,
            ),
            ParameterDefinition(
                symbol="a",
                unit="",
                description="Some exponent",
                value=0.5,
                lower_limit=0.0,
                upper_limit=1.0,
                fixed=False,
            ),
        ],
    ),
)


# # The new element should now be available for use like any of the
# # elements that are bundled with pyimpspec.
# circuit: Circuit = parse_cdc("Ude")

# circuit


def TLM_Z(
    omega: float | NDArray[float64],
    T: float64,
    i_0: float64,
    a: float64,
    geom_factor: float64,
    c_R: float64,
    D_R: float64,
    c_O: float64,
    D_O: float64,
    C_dl: float64,
    alpha: float64,
    rho_i: float64,
    rho_e: float64,
    l: float64,
    A: float64,
    A_s: float64,
) -> complex | NDArray[complex128]:
    """
    Calculate the Transmission Line Model (TLM) impedance for porous electrodes.

    Implements the transmission line model for porous electrodes including charge transfer,
    diffusion, and transport effects through the electrode structure.

    Parameters:
        omega (float | NDArray[float64]): Angular frequency in rad/s (radians per second).
        T (float64): Temperature in K (Kelvin).
        i_0 (float64): Exchange current density in A/cm² (amperes per square centimeter).
        a (float64): Nernstian diffusion layer thickness in cm (centimeters).
        geom_factor (float64): Geometry factor for pore shape and form (dimensionless).
        c_R (float64): Bulk concentration of reduced species in mol/cm³ (moles per cubic centimeter).
        D_R (float64): Diffusion coefficient of reduced species in cm²/s (square centimeters per second).
        c_O (float64): Bulk concentration of oxidized species in mol/cm³ (moles per cubic centimeter).
        D_O (float64): Diffusion coefficient of oxidized species in cm²/s (square centimeters per second).
        C_dl (float64): Area-specific double layer capacitance in F/cm² (farads per square centimeter).
        alpha (float64): CPE power factor (0 < alpha ≤ 1). alpha=1 for ideal capacitor.
        rho_i (float64): Ionic resistivity in Ω·cm (ohm-centimeters).
        rho_e (float64): Electronic resistivity in Ω·cm (ohm-centimeters).
        l (float64): Electrode thickness in cm (centimeters).
        A (float64): Geometric electrode area in cm² (square centimeters).
        A_s (float64): Total internal surface area in cm² (square centimeters).

    Returns:
        complex | NDArray[complex128]: Complex impedance Z(ω) in Ω (Ohms).

    Notes:
        Model accounts for:
        - Charge transfer resistance and kinetics
        - Diffusion impedance (Warburg-like) for redox species
        - Double layer capacitance (CPE behavior via alpha parameter)
        - Ionic transport through electrolyte phase
        - Electronic transport through electrode matrix
        Constants: R = 8.31 J/(mol·K), F = 96485.33212 C/mol
        Reference: Zawodzinski et al. DOI: 10.1149/2.045406jes
    """
    # Physical constants
    R = 8.31  # Gas constant (J/(mol·K)) = kg·m²·s⁻²·K⁻¹·mol⁻¹
    F = 9.64853321233100184e4  # Faraday constant (C/mol)

    # Convert to numpy array for consistent handling
    omega = np.asarray(omega)

    # Calculate surface-specific impedance Z_s_prime (Ohm·cm²)
    # Charge transfer + diffusion contributions
    Z_s_prime = (
        (R * T) / (F * i_0)
        + (R * T * a / (geom_factor * (F**2) * c_R * D_R))
        * np.tanh(a * np.sqrt(1j * omega / D_R))
        / (a * np.sqrt(1j * omega / D_R))
        + (R * T * a / (geom_factor * (F**2) * c_O * D_O))
        * np.tanh(a * np.sqrt(1j * omega / D_O))
        / (a * np.sqrt(1j * omega / D_O))
    )

    # Surface impedance including double layer capacitance (Ohm·cm²)
    Z_s = ((1 / Z_s_prime) + ((1j * omega) ** alpha) * C_dl) ** (-1)

    # Transmission line propagation constant
    psi = np.sqrt((A_s / Z_s) * (rho_i + rho_e) * l * A)
    # Total electrode impedance (Ohm)
    Z = (l / A) * (
        ((rho_i**2 + rho_e**2) / (rho_i + rho_e)) * (1 / (np.tanh(psi) * psi))
        + ((rho_i * rho_e) / (rho_i + rho_e)) * ((psi * np.sinh(psi)) ** -1 + 1)
    )

    return Z


def TLM_Z_2(
    omega: float | NDArray[float64],
    k_0: float64,
    a: float64,
    p: float64,
    v: float64,
    c_R: float64,
    D_R: float64,
    c_O: float64,
    D_O: float64,
    q_dl: float64,
    alpha: float64,
    rho_i: float64,
    rho_e: float64,
    l: float64,
    A: float64,
    A_s: float64,
    reflective_bc: bool = True,
    T: float64 = 298,
    n: int = 1,
    beta: float64 = 0.5,
) -> complex | NDArray[complex128]:
    """
    Calculate the Transmission Line Model (TLM) impedance for porous electrodes with flexible boundary conditions.

    Implements the transmission line model for porous electrodes with support for reflective and
    transmissive boundary conditions, including charge transfer, diffusion, and transport effects.

    Parameters:
        omega (float | NDArray[float64]): Angular frequency in rad/s (radians per second).
        k_0 (float64): Standard rate constant in cm/s (centimeters per second).
        a (float64): Nernstian diffusion layer thickness in cm (centimeters).
        p (float64): Pore size in cm (centimeters).
        v (float64): Fraction of surface with transmissive boundary condition (dimensionless, 0-1).
        c_R (float64): Bulk concentration of reduced species in mol/cm³ (moles per cubic centimeter).
        D_R (float64): Diffusion coefficient of reduced species in cm²/s (square centimeters per second).
        c_O (float64): Bulk concentration of oxidized species in mol/cm³ (moles per cubic centimeter).
        D_O (float64): Diffusion coefficient of oxidized species in cm²/s (square centimeters per second).
        q_dl (float64): Area-specific double layer capacitance in F/cm² (farads per square centimeter).
        alpha (float64): CPE power factor (0 < alpha ≤ 1). alpha=1 for ideal capacitor.
        rho_i (float64): Ionic resistivity in Ω·cm (ohm-centimeters).
        rho_e (float64): Electronic resistivity in Ω·cm (ohm-centimeters).
        l (float64): Electrode thickness in cm (centimeters).
        A (float64): Geometric electrode area in cm² (square centimeters).
        A_s (float64): Total internal surface area in cm² (square centimeters).
        reflective_bc (bool): If True, apply reflective boundary conditions; if False, transmissive only. Default: True.
        T (float64): Temperature in K (Kelvin). Default: 298 K.
        n (int): Number of electrons transferred. Default: 1.
        beta (float64): Charge transfer coefficient (0 < beta < 1). Default: 0.5.

    Returns:
        complex | NDArray[complex128]: Complex impedance Z(ω) in Ω (Ohms).

    Notes:
        Model accounts for:
        - Charge transfer resistance and kinetics
        - Semi-infinite diffusion impedance (Warburg-like) for redox species
        - Finite diffusion impedance with boundary condition options
        - Double layer capacitance (CPE behavior via alpha parameter)
        - Ionic and electronic transport through porous electrode matrix
        Constants: R = 8.31 J/(mol·K), F = 96485.33212 C/mol
        Reference: Zawodzinski et al. DOI: 10.1149/2.045406jes
    """
    # Physical constants
    R = 8.31  # Gas constant (J/(mol·K)) = kg·m²·s⁻²·K⁻¹·mol⁻¹
    F = 9.64853321233100184e4  # Faraday constant (C/mol)

    # Convert to numpy array for consistent handling
    omega = np.asarray(omega)

    # Calculate charge transfer impedance z_ctr (Ohm·cm²)
    z_ctr = (R * T) / (n**2 * F**2 * k_0 * (c_R**beta) * (c_O ** (1 - beta)))

    # Calculate semi-infinite diffusion impedance z_ws (Ohm·cm²)
    # Uses tanh for transmissive boundary condition
    sqrt_omega_DR = np.sqrt((a**2) * 1j * omega / D_R)
    sqrt_omega_DO = np.sqrt((a**2) * 1j * omega / D_O)

    z_ws = (R * T * a / (n**2 * F**2)) * (
        np.tanh(sqrt_omega_DR) / (c_R * D_R * sqrt_omega_DR)
        + np.tanh(sqrt_omega_DO) / (c_O * D_O * sqrt_omega_DO)
    )

    # Calculate finite diffusion impedance z_wo (Ohm·cm²)
    # Uses coth for reflective boundary condition
    p_half = p / 2
    sqrt_omega_DR_finite = np.sqrt((p_half**2) * 1j * omega / D_R)
    sqrt_omega_DO_finite = np.sqrt((p_half**2) * 1j * omega / D_O)

    z_wo = (R * T * p_half / (n**2 * F**2)) * (
        (1 / np.tanh(sqrt_omega_DR_finite)) / (c_R * D_R * sqrt_omega_DR_finite)
        + (1 / np.tanh(sqrt_omega_DO_finite)) / (c_O * D_O * sqrt_omega_DO_finite)
    )

    # Calculate combined diffusion impedance z_dif (Ohm·cm²)
    if reflective_bc:
        z_dif = ((v / (z_ws)) + ((1 - v) / z_wo)) ** (-1)
    else:
        # When reflective_bc is False, v acts as a geometry factor
        z_dif = z_ws / v

    # Surface impedance including double layer capacitance (Ohm·cm²)
    z_s = ((1 / (z_ctr + z_dif)) + ((1j * omega) ** alpha) * q_dl) ** (-1)

    # Transmission line propagation constant
    psi = np.sqrt((A_s / z_s) * (rho_i + rho_e) * l / A)

    # Total electrode impedance (Ohm)
    Z = (l / A) * (
        ((rho_i**2 + rho_e**2) / (rho_i + rho_e)) * (1 / (np.tanh(psi) * psi))
        + ((rho_i * rho_e) / (rho_i + rho_e)) * ((2 / (psi * np.sinh(psi))) + 1)
    )

    return Z
