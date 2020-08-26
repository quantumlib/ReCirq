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
import os
import shutil

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

# Certain notebooks are saved pre-executed. Keeping this environment variable
# as "auto" prevents re-executing (which takes time!). The CI will
# not re-execute these notebooks.
# Set this to "always" to execute all notebooks.
# https://github.com/quantumlib/ReCirq/issues/15
nbsphinx_execute = os.environ.get(
    'NBSPHINX_EXECUTE_NOTEBOOKS', 'auto')  # can also be "always" or "never".

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


def _rmtree_if_exists(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def env_before_read_docs(app, env, docnames):
    """Re-order `docnames` to respect the execution-order dependency
    of notebooks.

    Dependencies:
        Readout-Data-Collection <-- Readout-Analysis
        qaoa/Tasks-Tutorial <-+-- qaoa/Precomputed-Analysis
                              +-- qaoa/Landscape-Analysis

    Data:
        Readout-Data-Collection saves
            cirq-results/readout-scan/2020-02-tutorial

        qaoa/Tasks-Tutorial saves
            cirq-results/qaoa-problems/2020-03-tutorial
            cirq-results/qaoa-precomputation/2020-03-tutorial
            cirq-results/qaoa-precomputed/2020-03-tutorial
            cirq-results/qaoa-p1-landscape/2020-03-tutorial

    """
    print(docnames)

    def _order(docname):
        if docname == 'Readout-Data-Collection':
            # must come before others
            return -1

        if docname == 'qaoa/Tasks-Tutorial':
            # must come before others
            return -1
        return 0

    docnames.sort(key=_order)

    print(docnames)

    if ('Readout-Data-Collection' in docnames
            and 'Readout-Analysis' not in docnames):
        # Mark the analysis notebook as changed
        docnames.append('Readout-Analysis')

    if 'qaoa/Tasks-Tutorial' in docnames:
        if 'qaoa/Precomputed-Analysis' not in docnames:
            # Mark the analysis notebook as changed
            docnames.append('qaoa/Precomputed-Analysis')
        if 'qaoa/Landscape-Analysis' not in docnames:
            # Mark the analysis notebook as changed
            docnames.append('qaoa/Landscape-Analysis')

    print(docnames)

    if 'Readout-Data-Collection' in docnames:
        from recirq.readout_scan import tasks as rs_tasks
        _rmtree_if_exists(f'{rs_tasks.DEFAULT_BASE_DIR}/2020-02-tutorial')

    if 'qaoa/Tasks-Tutorial' in docnames and nbsphinx_execute == 'always':
        # Clear data that's already been collected, but only if we're actually
        # executing this notebook. This notebook is saved pre-executed, so
        # set environment variable NBSPHINX_EXECUTE_NOTEBOOKS to 'auto' to
        # prevent deleting the data and running from scratch (which can take
        # a long time).
        from recirq.qaoa.experiments import precomputed_execution_tasks as qaoa_pre_tasks
        _rmtree_if_exists(f'{qaoa_pre_tasks.DEFAULT_PROBLEM_GENERATION_BASE_DIR}/2020-03-tutorial')
        _rmtree_if_exists(f'{qaoa_pre_tasks.DEFAULT_PRECOMPUTATION_BASE_DIR}/2020-03-tutorial')
        _rmtree_if_exists(f'{qaoa_pre_tasks.DEFAULT_BASE_DIR}/2020-03-tutorial')

        from recirq.qaoa.experiments import p1_landscape_tasks as qaoa_p1_tasks
        _rmtree_if_exists(f'{qaoa_p1_tasks.DEFAULT_BASE_DIR}/2020-03-tutorial')


import subprocess

REPO_DIR = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                            stdout=subprocess.PIPE).communicate()[0].rstrip().decode('utf-8')


def setup(app):
    FROM = f'{REPO_DIR}/docs'
    TO = f'{REPO_DIR}/dev_tools/docs/sphinx'
    from distutils.dir_util import copy_tree  # allows overwriting
    copy_tree(f'{FROM}/tutorials', f'{TO}/tutorials')
    shutil.copy(f'{FROM}/images/g3618.png', f'{TO}/_static/')
    shutil.copy(f'{FROM}/images/recirq_logo_notext.png', f'{TO}/_static/')

    app.connect('env-before-read-docs', env_before_read_docs)
