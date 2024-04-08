# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
from pathlib import Path

modules_parent_folder = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, modules_parent_folder)

# -- Project information -----------------------------------------------------

project = 'statanalysis'
copyright = '2024, Hermann Agossou'
author = 'Hermann Agossou'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# The master toctree document.
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # like open3d:0.17
    # generate documentation for Python modules, classes, and functions, from docstring
    "sphinx.ext.autodoc",
    # generates summaries of modules, classes, and functions based on the documentation generated by sphinx.ext.autodoc
    "sphinx.ext.autosummary",
    # the Napoleon style of docstrings for Python modules, classes, and functions, is easy to read and write
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",  # display mathematical equations
    "sphinx.ext.todo",  # include "todo" items in your documentation
    # "nbsphinx", #include Jupyter Notebooks in your documentation
    "m2r2",  # convert Markdown files to reStructuredText (RST) format,
    # more from https://github.com/cimarieta/sphinx-autodoc-example
    # "nbsphinx", #include Jupyter Notebooks in your documentation
    # 'm2r2', #convert Markdown files to reStructuredText (RST) format,
    # more from https://github.com/cimarieta/sphinx-autodoc-example
    # 'sphinx.ext.doctest', # test code snippets in the documentation
    # 'sphinx.ext.intersphinx', # link to external documentation
    # 'sphinx.ext.coverage', # measure the coverage of the documentation
    # 'sphinx.ext.ifconfig', # include content based on configuration options
    # 'sphinx.ext.viewcode', # show the source code of modules and functions
    # 'sphinx.ext.githubpages', # publish documentation on GitHub Pages
]


# Napoleon settings
# from https://github.com/cimarieta/sphinx-autodoc-example
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'
html_theme = 'sphinxdoc'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
