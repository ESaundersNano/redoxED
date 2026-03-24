# if import this file, can use custom elements defined herein.

import numpy as np
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
    def _impedance(self, f: Frequencies, X: float, a: float) -> ComplexImpedances:
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

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
    T : float
        Temperature (K)
    i_0 : float
        Exchange current density (A/cm²)
    a : float
        Nernstian diffusion layer thickness (cm)
    geom_factor : float
        Geometry factor for shape and form of pores (dimensionless)
    c_R : float
        Bulk concentration of reduced species (mol/cm³)
    D_R : float
        Diffusion coefficient of reduced species (cm²/s)
    c_O : float
        Bulk concentration of oxidized species (mol/cm³)
    D_O : float
        Diffusion coefficient of oxidized species (cm²/s)
    C_dl : float
        Area-specific double layer capacitance (F/cm²)
    alpha : float
        CPE power factor (0 < alpha ≤ 1, alpha=1 for ideal capacitor)
    rho_i : float
        Ionic resistivity (electrolyte specific resistivity × porosity) (Ohm·cm)
    rho_e : float
        Electronic resistivity (electrode resistivity) (Ohm·cm)
    l : float
        Electrode thickness (cm)
    A : float
        Geometric area (cm²)
    A_s : float
        Total internal surface area of electrode (cm²)

    Returns
    -------
    complex or np.ndarray
        Complex impedance Z(ω) (Ohm)

    Notes
    -----
    Implements the Transmission Line Model (TLM) for porous electrodes.
    Based on the model developed by Zawodzinski et al.
    DOI: 10.1149/2.045406jes

    The model accounts for:
    - Charge transfer resistance
    - Diffusion impedance (Warburg-like) for both oxidized and reduced species
    - Double layer capacitance (with CPE behavior)
    - Ionic and electronic transport through the porous structure

    Physical constants used:
    - R = 8.31 J/(mol·K) - Universal gas constant
    - F = 9.64853321233100184e4 C/mol - Faraday constant

    The impedance Z_s represents the surface-specific impedance (Ohm·cm²),
    which is then combined with the transmission line effects to give
    the overall electrode impedance.
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
    Calculate the Transmission Line Model (TLM) impedance for porous electrodes.

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
    reflective_bc : bool
        If True, applies reflective boundary conditions at the pores as well as transmissive boundary.
    T : float
        Temperature (K)
    k_0 : float
        Standard rate constant (cm/s)
    n : int
        Number of electrons transferred
    beta : float
        Charge transfer coefficient (0 < beta < 1)
    a : float
        Nernstian diffusion layer thickness (cm)
    p : float
        Size of pores (cm)
    v : float
        Fraction of surface that receives transmissive boundary condition (dimensionless). When reflective_bc is False, acts as geometry factor fudging constant.
    c_R : float
        Bulk concentration of reduced species (mol/cm³)
    D_R : float
        Diffusion coefficient of reduced species (cm²/s)
    c_O : float
        Bulk concentration of oxidized species (mol/cm³)
    D_O : float
        Diffusion coefficient of oxidized species (cm²/s)
    q_dl : float
        Area-specific double layer capacitance (F/cm²)
    alpha : float
        CPE power factor (0 < alpha ≤ 1, alpha=1 for ideal capacitor)
    rho_i : float
        Ionic resistivity (electrolyte specific resistivity × porosity) (Ohm·cm)
    rho_e : float
        Electronic resistivity (electrode resistivity x porosity) (Ohm·cm)
    l : float
        Electrode thickness (cm)
    A : float
        Geometric area (cm²)
    A_s : float
        Total internal surface area of electrode (cm²)

    Returns
    -------
    complex or np.ndarray
        Complex impedance Z(ω) (Ohm)

    Notes
    -----
    Implements the Transmission Line Model (TLM) for porous electrodes, expanded for different boundary conditions.
    Based on the model developed by Zawodzinski et al.
    DOI: 10.1149/2.045406jes

    The model accounts for:
    - Charge transfer resistance
    - Diffusion impedance (Warburg-like) for both oxidized and reduced species
    - Double layer capacitance (with CPE behavior)
    - Ionic and electronic transport through the porous structure

    Physical constants used:
    - R = 8.31 J/(mol·K) - Universal gas constant
    - F = 9.64853321233100184e4 C/mol - Faraday constant

    The impedance Z_s represents the surface-specific impedance (Ohm·cm²),
    which is then combined with the transmission line effects to give
    the overall electrode impedance.
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
