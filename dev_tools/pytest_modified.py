import git
import pytest
import sys

from pathlib import Path


def get_subdir_for_pytest(path):
  if len(path.parts) >= 3 and path.parts[0] == 'recirq':
    # path is something inside of one of the subdirectories under 'recirq/'
    # return the top-level subdirectory inside of 'recirq'
    # For example, if path is 'recirq/foo/bar.txt' return 'recirq/foo'.
    return str(Path(path.parts[0:2]))
  if len(path.parts) == 2 and path.parts[0] == 'recirq':
    # path is something inside 'recirq', ex 'recirq/foo.txt'.
    # Conservatively assume all 'recirq' code may depend on the changes and run
    # all recirq pytests.
    return 'recirq'
  # path is something else not under the 'recirq' subdirectory.
  # Conservatively assume we should run all pytests.
  return '.'


def get_modified_subdirs():
  repo = git.Repo()
  modified_paths = list(Path(item.a_path) for item in repo.index.diff('HEAD'))
  modified_dirs = set(get_subdir_for_pytest(f) for f in modified_paths)
  return list(modified_dirs)


if __name__ == "__main__":
  modified_subdirs = get_modified_subdirs()
  if modified_subdirs:
    pytest_args = modified_subdirs + sys.argv[1:]
    print("executing pytest with args: " + ",".join(pytest_args))
    sys.exit(pytest.main(pytest_args))
