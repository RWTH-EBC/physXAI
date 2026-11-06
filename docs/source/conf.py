"""
Sphinx configuration of the physXAI documentation.

The documentation is built by the EBC CI template (RWTH-EBC/ci_templates, sphinx_doc.yml):

    sphinx-apidoc -M -o docs/source/code physXAI
    sphinx-build -b html docs/source docs_output

Locally (with `pip install -e .[docs]`) the same commands work; `-W --keep-going` turns warnings into errors.
The user guide is written in Markdown (MyST), the API reference is generated from the Google style docstrings.
"""

from importlib.metadata import version as _version

project = "physXAI"
author = "RWTH Aachen University, E.ON Energy Research Center, Institute for Energy Efficient Buildings and Indoor Climate"
copyright = "RWTH Aachen University, E.ON ERC EBC"
release = _version("physXAI")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.autodoc_pydantic",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build"]
templates_path = []

# Markdown (MyST)
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3  # links like [text](evaluation.md#comparing-models)

# API reference
autodoc_member_order = "bysource"
# package pages document only what is defined in the module (no duplicates of re-exported names)
autodoc_default_options = {"members": True, "show-inheritance": True, "ignore-module-all": True}
# docstrings of keras / sklearn base classes are not part of the physXAI documentation
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True  # "Attributes:" sections do not duplicate the attributes documented by autodoc
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
# `name` in docstrings (Markdown habit) is rendered as code instead of a title reference
default_role = "code"
# heavy dependencies are imported, they are installed with physXAI
nitpicky = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_title = f"physXAI {release}"
html_show_sourcelink = False
