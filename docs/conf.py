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
from datetime import date
from bosk import __version__ as bosk_version
from pathlib import Path
from shutil import copy
sys.path.insert(0, os.path.abspath('..'))


# -- Maintain jupyter notebooks folder --------------------------------

EXAMPLES_DIR = '../examples'
OUTPUT_NOTEBOOKS_DIR = 'notebooks/'

def clean_notebooks(dir: str = OUTPUT_NOTEBOOKS_DIR) -> None:
    examples_path = Path(dir)
    for file in examples_path.glob('**/*.ipynb'):
        os.remove(str(file))

def copy_example_notebooks(inp_dir: str = EXAMPLES_DIR, out_dir: str = OUTPUT_NOTEBOOKS_DIR) -> None:
    examples_path = Path(inp_dir)
    for file in examples_path.glob('**/*.ipynb'):
        copy(str(file), out_dir)

clean_notebooks()
copy_example_notebooks()

# -- Project information -----------------------------------------------------

project = 'bosk'
copyright = f'2022-{date.today().year}, NTAILab'
author = 'NTAILab'

# The full version, including alpha/beta/rc tags
release = str(bosk_version)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.coverage',
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'autoapi.extension',
    'nbsphinx',
]
autoapi_dirs = ['../bosk']
autoapi_add_toctree_entry = True
autoapi_generate_api_docs = True
autoapi_keep_files = True
autodoc_typehints = 'description'
# autoapi_python_class_content = 'init'
autodoc_default_flags = ['undoc-members', 'inherited-members']
autodoc_default_options = {
    'special-members': '__init__, __repr__',
    'undoc-members': True,
    'inherited-members': True,
}


# def skip(app, what, name, obj, would_skip, options):
#     if name.startswith("__") and not name.endswith("__"):
#         return True
#     return would_skip
#
# def setup(app):
#     app.connect("autodoc-skip-member", skip)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
