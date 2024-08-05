# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'vindy')))
# sys.path.insert(0, os.path.abspath(os.path.join('..', 'vindy')))
src_path = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, src_path)
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'vindy', 'distributions')))
print(src_path)

project = 'VENI, VINDy, VICI'
copyright = '2024, Jonas Kneifl, Paolo Conti'
author = 'Jonas Kneifl, Paolo Conti'
release = '0.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo",
              "sphinx.ext.viewcode",
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              # 'numpydoc',
              'myst_parser']

templates_path = ['_templates']
exclude_patterns = []

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Numpydoc settings
numpydoc_show_class_members = False