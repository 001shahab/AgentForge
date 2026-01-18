"""
Sphinx configuration for AgentForge documentation.

I've set up Sphinx to automatically generate API documentation from
our docstrings. Run `make html` in the docs folder to build.

Author: Prof. Shahab Anbarjafari
"""

import os
import sys

# Add the project root to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'AgentForge'
copyright = '2024, Prof. Shahab Anbarjafari, 3S Holding OÜ'
author = 'Prof. Shahab Anbarjafari'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google-style docstrings
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

