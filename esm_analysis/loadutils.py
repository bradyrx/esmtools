"""Contains functions for loading sample datasets and region masks."""

import os as _os
from urllib.request import urlretrieve as _urlretrieve
from xarray.backends.api import open_dataset as _open_dataset
import hashlib

_default_cache_dir = _os.sep.join(('~', '.esm_analysis_data'))


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def open_dataset(
    name,
    cache=True,
    cache_dir=_default_cache_dir,
    github_url='https://github.com/bradyrx/climdata',
    branch='master',
    extension=None,
    **kws
):
    """Load example data or a mask from an online repository.

    This is a function from `xarray.tutorial` to load an online dataset
    with minimal package imports. I am copying it here because it looks like
    it will soon be deprecated. Also, I've added the ability to point to
    data files that are not in the main folder of the repo (i.e., they are
    in subfolders).

    Note that this requires an md5 file to be loaded. Check the github
    repo bradyrx/climdata for a python script that converts .nc files into
    md5 files.

    Args:
        name: (str) Name of the netcdf file containing the dataset, without
              the .nc extension.
        cache_dir: (str, optional) The directory in which to search
                   for and cache the data.
        cache: (bool, optional) If true, cache data locally for use on later
               calls.
        github_url: (str, optional) Github repository where the data is stored.
        branch: (str, optional) The git branch to download from.
        extension: (str, optional) Subfolder within the repository where the
                   data is stored.
        kws: (dict, optional) Keywords passed to xarray.open_dataset

    Returns:
        The desired xarray dataset.
    """
    if name.endswith('.nc'):
        name = name[:-3]
    longdir = _os.path.expanduser(cache_dir)
    fullname = name + '.nc'
    localfile = _os.sep.join((longdir, fullname))
    md5name = name + '.md5'
    md5file = _os.sep.join((longdir, md5name))

    if not _os.path.exists(localfile):
        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        if extension is not None:
            url = '/'.join((github_url, 'raw', branch, extension, fullname))
            _urlretrieve(url, localfile)
            url = '/'.join((github_url, 'raw', branch, extension, md5name))
            _urlretrieve(url, md5file)
        else:
            url = '/'.join((github_url, 'raw', branch, fullname))
            _urlretrieve(url, localfile)
            url = '/'.join((github_url, 'raw', branch, md5name))
            _urlretrieve(url, md5file)

        localmd5 = file_md5_checksum(localfile)
        with open(md5file, 'r') as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            _os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise IOError(msg)

    ds = _open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        _os.remove(localfile)

    return ds
