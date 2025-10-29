# if import this file, can use custom elements defined herein.

import pyimpspec
import numpy as np
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
from numpy import pi, inf

import pyimpspec

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
