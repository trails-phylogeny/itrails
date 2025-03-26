# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
import os
import re
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

# Strip off any Git commit or local metadata using regex
# e.g. "0.1.0a73.dev1+ge16dbd8" → "0.1.0a73"
base_version_match = re.match(r"^([0-9a-zA-Z.\-]+)", __version__)
clean_version = base_version_match.group(1) if base_version_match else "dev"

release = clean_version  # full version, cleaned
version = ".".join(clean_version.split(".")[0:2])  # e.g. "0.1"

html_title = f"itrails v{release} documentation"
html_short_title = f"itrails v{release}"


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
html_css_files = ["custom.css"]


def setup(app):
    app.add_js_file("rtd_version.js")
