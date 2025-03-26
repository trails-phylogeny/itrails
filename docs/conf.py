# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os
import sys

# Add your package to sys.path
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "itrails"
authors = ["David Martin-Pestana", "Iker Rivas-González"]
author = ", ".join(authors)
copyright = f"{datetime.datetime.now().year}, {author}"

# -- Version handling -------------------------------------------------------

try:
    from itrails import __version__
except ImportError:
    __version__ = "unknown"

release = __version__
version = ".".join(__version__.split(".")[:2]) if __version__ != "unknown" else "dev"

# This allows you to use |release| in your .rst files
rst_epilog = f"\n.. |release| replace:: {release}\n"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Theme
html_theme = "furo"
templates_path = ["_templates"]
exclude_patterns = []
html_static_path = ["_static"]
