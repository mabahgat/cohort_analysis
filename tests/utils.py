from pathlib import Path


def get_abs_file_path(file, relative_path: str):
    """
    Gets absolute file path for a relative path to a given file
    :param file: caller file
    :param relative_path: relative path string
    :return: absolute path string
    """
    return Path(file).parent.resolve() / relative_path
