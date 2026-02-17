# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'vindy')))
src_path = os.path.abspath(os.path.join('..', '..', 'vindy'))
# src_path = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, src_path)
# sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'vindy', 'distributions')))


# Ensure local package root is on sys.path so Sphinx imports the local source, not an installed package
project_root = os.path.abspath(os.path.join('..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VENI-VINDy-VICI'
copyright = '2026, Jonas Kneifl, Paolo Conti'
author = 'Jonas Kneifl, Paolo Conti'
release = '2026'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo",
              "sphinx.ext.viewcode",
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'numpydoc',
              'myst_parser'
              ]

templates_path = ['_templates']
exclude_patterns = []

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Mock heavy or optional dependencies during doc builds so 'make html' works
# in environments that don't have the full ML stack installed.
autodoc_mock_imports = [
    'tensorflow',
    'tensorflow.keras',
    'tensorflow_probability',
    'tensorflow_model_optimization',
    'numpy',
    'scipy',
    'matplotlib',
    'matplotlib.pyplot',
    'sympy',
    'imageio',
    # Also mock keras and optree to avoid import-time issues when Keras/TF are present but incompatible
    'keras',
    'optree',
    'optree.registry',
    'optree.tree_api',
]

# Try to use the preferred theme, but fall back to a built-in theme if it's missing.
try:
    import sphinx_rtd_theme  # noqa: F401
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'alabaster'

html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Include __init__ docstring with class doc
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_typehints = 'both'

# Numpydoc settings
numpydoc_show_class_members = False

# Ensure autodoc presents class __init__ signature on the class page
autoclass_content = 'both'
# When possible, use the __init__ signature as the class signature
autodoc_class_signature = 'mixed'

# Optionally show all members by default (use with care)
# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#}
