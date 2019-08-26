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
import os
import sys
sys.path.insert(0, os.path.abspath('../dataflame'))


# -- Project information -----------------------------------------------------

project = 'DataFlame'
copyright = '2019, Shayan Fazeli'
author = 'Shayan Fazeli'

# The full version, including alpha/beta/rc tags
release = '1.0.0-beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    #'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    #'numpydoc',
    'sphinx_paramlinks'
    #'rst2pdf.pdfbuilder'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

import sphinxbootstrap4theme

html_theme = 'sphinxbootstrap4theme'
html_theme_path = [sphinxbootstrap4theme.get_path()]

# Html logo in navbar.
# Fit in the navbar at the height of image is 37 px.
html_logo = '_static/logo.jpg'

html_theme_options = {
    # Navbar style.
    # Values: 'fixed-top', 'full' (Default: 'fixed-top')
    'navbar_style' : 'fixed-top',

    # Navbar link color modifier class.
    # Values: 'dark', 'light' (Default: 'dark')
    'navbar_color_class' : 'dark',

    # Navbar background color class.
    # Values: 'inverse', 'primary', 'faded', 'success',
    #         'info', 'warning', 'danger' (Default: 'inverse')
    'navbar_bg_class' : 'inverse',

    # Show global TOC in navbar.
    # To display up to 4 tier in the drop-down menu.
    # Values: True, False (Default: True)
    'navbar_show_pages' : True,

    # Link name for global TOC in navbar.
    # (Default: 'Pages')
    'navbar_pages_title' : 'Pages',

    # Specify a list of menu in navbar.
    # Tuples forms:
    #  ('Name', 'external url or path of pages in the document', boolean)
    # Third argument:
    # True indicates an external link.
    # False indicates path of pages in the document.
    'navbar_links' : [
         ('Home', 'index', False),
         ("Link", "http://example.com", True)
    ],

    # Total width(%) of the document and the sidebar.
    # (Default: 80%)
    'main_width' : '80%',

    # Render sidebar.
    # Values: True, False (Default: True)
    'show_sidebar' : True,

    # Render sidebar in the right of the document.
    # Values：True, False (Default: False)
    'sidebar_right': False,

    # Fix sidebar.
    # Values: True, False (Default: True)
    'sidebar_fixed': True,

    # Html table header class.
    # Values: 'inverse', 'light' (Deafult: 'inverse')
    'table_thead_class' : 'inverse'
}

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp'
}