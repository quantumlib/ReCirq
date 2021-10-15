import git
import pytest
import sys

from pathlib import Path
from typing import List


def get_subdir_for_pytest(path: Path) -> str:
    """Returns the subdir to be pytest'd when a file is modified.

    Args:
      path: the Path object for the modified file
    Returns:
      the subdirectory path prefix string
    """
    if len(path.parts) >= 3 and path.parts[0] == "recirq":
        # path is something inside of one of the subdirectories under 'recirq/'
        # return the top-level subdirectory inside of 'recirq'
        # For example, if path is 'recirq/foo/bar.txt' return 'recirq/foo'.
        return str(Path(*path.parts[0:2]))
    if len(path.parts) == 2 and path.parts[0] == "recirq":
        # path is something inside 'recirq', ex 'recirq/foo.txt'.
        # Conservatively assume all 'recirq' code may depend on the changes and run
        # all recirq pytests.
        return "recirq"
    # path is something else not under the 'recirq' subdirectory.
    # Conservatively assume we should run all pytests.
    return "."


def get_modified_subdirs_for_pytest(repo_path: str) -> List[str]:
    """Returns subdirectories for pytesting that contain git modifications.

    Args:
      repo_path: path to the git repository which will be examined for diffs
    Returns:
      a (possibly empty) list of subdirectory strings containing modifications
    """
    repo = git.Repo(repo_path)
    modified_paths = list(Path(item.a_path) for item in repo.index.diff("HEAD"))
    modified_dirs = set(get_subdir_for_pytest(f) for f in modified_paths)
    return list(modified_dirs)


if __name__ == "__main__":
    modified_subdirs = get_modified_subdirs_for_pytest(".")
    if modified_subdirs:
        # Allow passing through additional argv to pytest.
        pytest_args = modified_subdirs + sys.argv[1:]
        print(f"executing pytest with args: [{', '.join(pytest_args)}]")
        sys.exit(pytest.main(pytest_args))
