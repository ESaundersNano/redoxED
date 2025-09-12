# redoxED

A Python package for electrochemical data analysis and visualization.

## Installation

To be added

## LaTeX Rendering

redoxED supports LaTeX rendering for publication-quality plots with mathematical notation. 

### Prerequisites

**For LaTeX rendering to work properly, you need a complete LaTeX distribution installed on your system.** We recommend installing **TeX Live** (cross-platform) or **MiKTeX** (Windows).

- **Linux**: Install TeX Live via your package manager: `sudo apt-get install texlive-full`
- **macOS**: Install MacTeX (TeX Live for Mac): https://www.tug.org/mactex/
- **Windows**: Install TeX Live or MiKTeX: https://www.tug.org/texlive/ or https://miktex.org/

### Global LaTeX Configuration

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

### Per-Plot LaTeX Control

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

### Important Notes

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
