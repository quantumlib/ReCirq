# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project = 'ReCirq'
copyright = '2020, Google Quantum'
author = 'Google Quantum'

extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon'
]

nbsphinx_allow_errors = False
nbsphinx_timeout = -1  # no timeout
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False

add_module_names = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store',
                    '**.ipynb_checkpoints', 'appengine']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_favicon = 'favicon.ico'
html_logo = '_static/recirq_logo_notext.png'


def env_before_read_docs(app, env, docnames):
    """Re-order `docnames` to respect the execution-order dependency
    of notebooks.

    Dependencies:
        Readout-Data-Collection <-- Readout-Analysis
    """

    def _order(docname):
        if docname == 'Readout-Data-Collection':
            # must come before others
            return -1
        return 0

    docnames.sort(key=_order)

    if ('Readout-Data-Collection' in docnames
            and 'Readout-Analysis' not in docnames):
        # Mark the analysis notebook as changed
        docnames.append('Readout-Analysis')


def setup(app):
    app.connect('env-before-read-docs', env_before_read_docs)
