# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# to see how to set up the path checkout answer with title "solution" here:
# https://stackoverflow.com/questions/10324393/sphinx-build-fail-autodoc-cant-import-find-module
sys.path.insert(0, os.path.abspath('../..'))

project = 'Compact Poisson Filters'
copyright = '2023, Ubisoft'
author = 'Shahin (Amir Hossein) Rabbani'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinxcontrib.video',
    # 'sphinx.ext.autosectionlabel'
]
napoleon_google_docstring = False
show_authors = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# --- Theme
# html_theme = 'bizstyle'
import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
