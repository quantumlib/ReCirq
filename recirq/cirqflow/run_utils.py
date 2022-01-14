import datetime
import os


def get_unique_run_id(fmt: str = 'run-{i}', base_data_dir: str = '.') -> str:
    """Find an unused run_id by checking for existing paths and incrementing `i` in `fmt` until
    an unused path name is found.

    Args:
        fmt: A format string containing {i} and optionally {date} to template trial run_ids.
        base_data_dir: The directory to check for the existence of files or directories.
    """
    i = 1
    while True:
        run_id = fmt.format(i=i, date=datetime.date.today().isoformat())
        if not os.path.exists(f'{base_data_dir}/{run_id}'):
            break  # found an unused run_id
        i += 1

    return run_id
