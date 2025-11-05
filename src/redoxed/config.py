"""
Configuration module for redoxED package.

This module provides global configuration settings that can be used
across the entire package, particularly for controlling LaTeX rendering
in plots.

Important Notes:
- LaTeX rendering requires a full LaTeX distribution (e.g., TeX Live)
- Do not mix usetex=True and usetex=False plots in the same Python session
  as matplotlib's global settings will cause font inconsistencies
- Set the global LaTeX preference once at the beginning of your session
"""

# Global configuration variables
USE_LATEX: bool = False
QP_SOLVER: str = "cvxopt"  # Options: "cvxopt" (default), "kvxopt"


def set_latex_mode(use_latex: bool) -> None:
    """
    Set global LaTeX rendering preference for all plots.

    Args:
        use_latex (bool): Whether to enable LaTeX rendering globally.

    Example:
        >>> from redoxed import config
        >>> config.set_latex_mode(True)  # Enable LaTeX for all plots
        >>> config.set_latex_mode(False) # Disable LaTeX for all plots
    """
    global USE_LATEX
    USE_LATEX = use_latex


def set_qp_solver(solver: str) -> None:
    """
    Set the QP solver backend.

    Args:
        solver (str): "cvxopt" or "kvxopt"
    """
    global QP_SOLVER
    if solver not in ("cvxopt", "kvxopt"):
        raise ValueError("QP_SOLVER must be 'cvxopt' or 'kvxopt'")
    QP_SOLVER = solver


def get_config() -> dict:
    """
    Get current configuration settings.

    Returns:
        dict: Dictionary containing current configuration.
    """
    return {
        "USE_LATEX": USE_LATEX,
        "QP_SOLVER": QP_SOLVER,
    }


def reset_config() -> None:
    """Reset all configuration to default values."""
    global USE_LATEX, QP_SOLVER
    USE_LATEX = False
    QP_SOLVER = "cvxopt"
