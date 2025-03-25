# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # Adjust to locate your package

try:
    from itrails import __version__
except ImportError:
    __version__ = "unknown"

# Use the package version for the docs.
release = __version__
# Optionally, you can define a shorter version (e.g., major.minor)
version = ".".join(__version__.split(".")[:2])
rst_epilog = f"\n.. |release| replace:: {release}\n"

project = "itrails"
copyright = "2025, David Martin-Pestana, Iker Rivas-González"
author = "David Martin-Pestana, Iker Rivas-González"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google or NumPy style docstrings
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
}

html_static_path = ["_static"]
