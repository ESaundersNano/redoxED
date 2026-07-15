# redoxED

A Python package for electrochemical data analysis and visualisation.

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/redoxED?logo=python&logoColor=white&v=1)](https://pypi.org/project/redoxED/)
[![GitHub License](https://img.shields.io/github/license/ESaundersNano/redoxED?logo=github&v=1)](https://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI - Version](https://img.shields.io/pypi/v/redoxED?logo=pypi&logoColor=white&v=1)](https://pypi.org/project/redoxED/)
<!-- Add in similar for tests and build? -->


## Table of contents

- [Installation](#installation)
- [About](#about)
- [Contributing](#citingcontributing)
- [Changelog](#changelog)
- [License](#license)

## Installation

The simplest terminal installation is to simply run:

```console
pip install redoxED
```

Alternatively, to use an editable local repository, clone this repository, navigate to the directory with pyproject.toml in terminal and use:

```console
pip install -e .
```

## About

See [notebooks](https://github.com/ESaundersNano/redoxED/tree/main/notebooks) for demonstration of main features.

### Echem Data Extraction
Battery cycling data, EIS, and polarisation data can be loaded directly from raw potentiostat files or csv. Alternatively, the classes defined for each data type can be instanced directly. 

### DRT Analysis
An open, transparent, and configurable implementation of distribution of relaxation times (DRT) analysis developed to support our work.

DRT is calculated from EIS spectra, based on code from https://github.com/ciuccislab/pyDRTtools and https://github.com/vyrjana/pyimpspec. The main distinction from theses solvers is the ability to explicitly choose the time constants used for calculation, independent of the frequency points provided. By default, our solver uses (1/2πf) vs (1/f) used in these other implmentations. 
DRT with lumped series capacitance is also implemented to support Li work. Additionally, all solver options (e.g. integration and optimisation algorithm choices) are made available for tuning by the user. Notably, the regularisation parameter optimisation algorithm was found to be important to prevent λ from sticking to initial guesses. 

Additional functionality allows for peak fitting and reconstructing impedance spectra. 

This module uses Tikhonov regularization and either radial basis function or piecewise linear discretization
- 10.1016/j.electacta.2015.09.097 # radial basis functions for DRT
- 10.1149/1945-7111/acbca4 # hyperparameter selection
- 10.1021/acselectrochem.5c00334 # DRTtools

### Plotting

redoxED supports batch plotting of DC and AC figures including:
- Battery efficiency plots
- Other battery metrics
- Polarisation curves
- Nyquist and Bode plots
- DRT spectra
- Residuals plots for model fits.

redoxED supports LaTeX rendering for publication-quality plots with mathematical notation. 

#### Prerequisites

**For LaTeX rendering to work properly, you need a complete LaTeX distribution installed on your system.** We recommend installing **TeX Live** (cross-platform) or **MiKTeX** (Windows).

- **Linux**: Install TeX Live via your package manager: `sudo apt-get install texlive-full`
- **macOS**: Install MacTeX (TeX Live for Mac): https://www.tug.org/mactex/
- **Windows**: Install TeX Live or MiKTeX: https://www.tug.org/texlive/ or https://miktex.org/

#### Global LaTeX Configuration

You can enable or disable LaTeX rendering globally for all plots:

```python
from redoxed import config

# Enable LaTeX for all plots (requires LaTeX installation)
config.set_latex_mode(True)

# Disable LaTeX for all plots (uses mathtext)
config.set_latex_mode(False)

# Check current setting
print(config.get_config())
```

#### Per-Plot LaTeX Control

You can override the global setting for individual plots:

```python
from redoxed.plots import NyquistPlot

# Use global LaTeX setting
plot1 = NyquistPlot()

# Override to disable LaTeX for this plot only
plot2 = NyquistPlot(usetex=False)

# Override to enable LaTeX for this plot only
plot3 = NyquistPlot(usetex=True)
```

#### Important Notes

⚠️ **Mixed LaTeX Usage Warning**: Do not mix `usetex=True` and `usetex=False` plots in the same Python session/notebook cell, as matplotlib's global settings will cause one to overwrite the other's font settings. Set the global configuration once at the beginning of your session.

✅ **Recommended Approach**:
```python
# At the beginning of your notebook/script
from redoxed import config
config.set_latex_mode(True)  # Set once globally

# All subsequent plots will use LaTeX
from redoxed.plots import NyquistPlot, DRTPlot
plot1 = NyquistPlot()
plot2 = DRTPlot()
```

❌ **Avoid This**:
```python
# Don't mix LaTeX settings in the same session
plot1 = NyquistPlot(usetex=True)   # Uses LaTeX
plot2 = NyquistPlot(usetex=False)  # Font settings may be inconsistent
```

## Citing/Contributing
This code was developed primarily for the PhD of Edward Saunders and further active development is not currently planned. 
Under the license, others are free to install the library, clone the repo for their own use, or duplicate the repo to make their own version if they appropriately credit this work.  

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for details.

## License

Copyright 2026 Edward Saunders

redoxED is licensed under the [GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.html).
