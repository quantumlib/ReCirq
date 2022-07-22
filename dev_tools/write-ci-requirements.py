import pathlib
import re
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List

REPO_DIR = pathlib.Path(__file__).parent.parent.resolve()
print('Repo dir:', REPO_DIR)

CIRQ_VERSIONS = {
    'previous': '==0.14.0',
    'current': '==0.15.0',
    'next': '>=1.0.0.dev',
}
"""Give names to relative Cirq versions so CI can have consistent names while versions 
get incremented."""


def _parse_requirements(path: pathlib.Path):
    """Read and strip comments from a requirements.txt-like file. """
    lines = [line.strip() for line in path.read_text().splitlines() if line]
    return [line for line in lines if not line.startswith('#')]


def _remove_version_spec(req: str) -> str:
    """Remove a version specifier like ==1.3 from a requirements.txt line."""
    components = re.split(r'>=|~=|<=|==', req)
    return components[0]


def _set_cirq_version(core_reqs: List[str], relative_cirq_version: str) -> List[str]:
    """Return a new version of `core_reqs` that pins cirq-like packages to the desired version."""
    cirq_version = CIRQ_VERSIONS[relative_cirq_version]
    to_change = 'cirq', 'cirq-google', 'cirq-core'

    new_reqs = []
    for req in core_reqs:
        without_spec = _remove_version_spec(req)
        if without_spec in to_change:
            new_reqs.append(f'{without_spec}{cirq_version}')
        else:
            new_reqs.append(req)

    return new_reqs


def _set_qaoa_hacks(qaoa_reqs: List[str], relative_cirq_version: str) -> List[str]:
    """Pytket pins to a specific cirq version and doesn't work with cirq "next".
    """
    if relative_cirq_version != 'next':
        return qaoa_reqs

    new_reqs = []
    for req in qaoa_reqs:
        without_spec = _remove_version_spec(req)
        if without_spec == 'pytket-cirq':
            continue
        else:
            new_reqs.append(req)

    return new_reqs


def main(*, out_fn: str = 'ci-requirements.txt', relative_cirq_version: str = 'current',
         all_extras: bool = False):
    """Write a requirements.txt file for CI installation and testing.

    Args:
         out_fn: The output filename
         relative_cirq_version: Pin the desired cirq version to either "current", "previous",
            or "next" version.
        all_extras: Whether to include all the extras_require dependencies.
    """
    core_reqs = _parse_requirements(REPO_DIR / 'requirements.txt')
    core_reqs = _set_cirq_version(core_reqs, relative_cirq_version)

    extras_require = [
        'otoc', 'qaoa', 'optimize', 'hfvqe', 'fermi_hubbard', 'qml_lfe'
    ]
    extras_require = {
        r: _parse_requirements(pathlib.Path(REPO_DIR / f'recirq/{r}/extra-requirements.txt'))
        for r in extras_require
    }
    extras_require['qaoa'] = _set_qaoa_hacks(extras_require['qaoa'], relative_cirq_version)

    lines = ['# Core requirements'] + core_reqs
    if all_extras:
        for name, reqs in extras_require.items():
            lines += ['', f'# {name}']
            lines += reqs
    lines += ['']

    out = '\n'.join(lines)
    sys.stdout.write(out)
    (REPO_DIR / out_fn).write_text(out)


def parse() -> Dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument('--out-fn', default='ci-requirements.txt')
    parser.add_argument('--relative-cirq-version', default='current')
    parser.add_argument('--all-extras', action='store_true', default=False)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**parse())
