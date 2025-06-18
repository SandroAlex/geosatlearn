import s3fs


def list_folders(base_folder: str) -> list:
    """
    List all folders in the base folder.

    Parameters
    ----------
    base_folder : str
        The S3 bucket folder where folders are stored.

    Returns
    -------
    list
        A list of dataset names (folders) in the specified S3 bucket folder.
    """

    fs = s3fs.S3FileSystem(anon=False)
    folders = fs.ls(base_folder)
    return folders
