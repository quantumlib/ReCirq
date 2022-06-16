import os
import shutil
import time
from multiprocessing import Pool
from subprocess import run as _run, CalledProcessError

BASE_PACKAGES = [
    # for running the notebooks
    "jupyter",
    # assumed to be part of colab
    "seaborn~=0.11.1",
]


def _check_notebook(notebook_fn: str, notebook_id: str, stdout, stderr):
    """Helper function to actually do the work in `check_notebook`.

    `check_notebook` has all the context managers and exception handling,
    which would otherwise result in highly indented code.

    Args:
        notebook_fn: The notebook filename
        notebook_id: A unique string id for the notebook that does not include `/`
        stdout: A file-like object to redirect stdout
        stderr: A file-like object to redirect stderr
    """
    print(f'Starting {notebook_id}')

    def run(*args, **kwargs):
        return _run(*args, check=True, stdout=stdout, stderr=stderr, **kwargs)

    # 1. create venv
    venv_dir = os.path.abspath(f'./notebook_envs/{notebook_id}')
    run(['python', '-m', 'venv', '--clear', venv_dir])

    # 2. basic colab-like environment
    pip = f'{venv_dir}/bin/pip'
    run([pip, 'install'] + BASE_PACKAGES)

    # 3. execute
    jupyter = f'{venv_dir}/bin/jupyter'
    env = os.environ.copy()
    env['PATH'] = f'{venv_dir}/bin:{env["PATH"]}'
    run([jupyter, 'nbconvert', '--to', 'html', '--execute', notebook_fn], cwd='../', env=env)

    # 4. clean up
    shutil.rmtree(venv_dir)


def check_notebook(notebook_fn: str):
    """Check a notebook.

     1. Create a venv just for that notebook
     2. Verify the notebook executes without error (and that it installs its own dependencies!)
     3. Clean up venv dir

    A scratch directory dev_tools/notebook_envs will be created containing tvenv as well as
    stdout and stderr logs for each notebook. Each of these files and directories will be
    named according to the "notebook_id", which is `notebook_fn.replace('/', '-')`.

    The executed notebook will be rendered to html alongside its original .ipynb file for
    spot-checking.

    Args:
        notebook_fn: The filename of the notebook relative to the repo root.
    """
    notebook_id = notebook_fn.replace('/', '-')
    start = time.perf_counter()
    with open(f'./notebook_envs/{notebook_id}.stdout', 'w') as stdout, \
            open(f'./notebook_envs/{notebook_id}.stderr', 'w') as stderr:
        try:
            _check_notebook(notebook_fn, notebook_id, stdout, stderr)
        except CalledProcessError:
            print('ERROR!', notebook_id)
    end = time.perf_counter()
    print(f'{notebook_id} {end - start:.1f}s')


NOTEBOOKS = [
    'docs/otoc/otoc_example.ipynb',
    'docs/guide/data_analysis.ipynb',
    'docs/guide/data_collection.ipynb',
    'docs/qaoa/example_problems.ipynb',
    'docs/qaoa/precomputed_analysis.ipynb',
    'docs/qaoa/hardware_grid_circuits.ipynb',
    'docs/qaoa/optimization_analysis.ipynb',
    'docs/qaoa/tasks.ipynb',
    'docs/qaoa/landscape_analysis.ipynb',
    'docs/qaoa/routing_with_tket.ipynb',
    'docs/hfvqe/molecular_data.ipynb',
    'docs/hfvqe/quickstart.ipynb',
    'docs/fermi_hubbard/publication_results.ipynb',
    'docs/fermi_hubbard/experiment_example.ipynb',
]


def main():
    os.chdir(os.path.dirname(__file__))
    os.makedirs('./notebook_envs', exist_ok=True)
    with Pool(4) as pool:
        results = pool.map(check_notebook, NOTEBOOKS)
    print(results)


if __name__ == '__main__':
    main()
