"""
Path and file operation tools.
"""

import platform
import os
import sys
import pathlib

PLATFORM = platform.system()
HOSTNAME = platform.node()

blackie = {
    'env': {'FREESURFER_HOME': '/usr/local/freesurfer',
            'SUBJECTS_DIR': '/mnt/d/CloudSynology/subjects',
            'setup_cmd': ' export PATH=$PATH:/usr/local/freesurfer/bin && export FREESURFER_HOME=/usr/local/freesurfer  && source $FREESURFER_HOME/SetUpFreeSurfer.sh'
            },
    'tools': {
        'dicoms_dir': '/mnt/d/CloudSynology/rawdata/__subj__/',
        'fs_dir': '/mnt/d/CloudSynology/subjects/__subj__/',
    }
}

osboxes = {
    'env': {'FREESURFER_HOME': '/usr/local/freesurfer',
            'SUBJECTS_DIR': '/home/osboxes/subjects',
            'setup_cmd': ' export PATH=$PATH:/usr/local/freesurfer/bin && export FREESURFER_HOME=/usr/local/freesurfer  && source $FREESURFER_HOME/SetUpFreeSurfer.sh'
            },
    'tools': {
        'dicoms_dir': '/home/osboxes/host/CloudSynology/rawdata/__subj__/',
        'fs_dir': '/home/osboxes/subjects/__subj__/',
    }
}


def wsl_tempfile(filename):
    """Creates a temporary filename in the current user's folder for native system and
    Windows Subsytem for Linux is native is Windows.

    Parameters
    ----------
    filename: string
        Name of temporary file

    Returns
    -------
    native_file: string
        Path to temporary file on native file system
    was_file: string
        Path to temporary file on WSL file system (ex: /mnt/c/Users/user/filename)
    """
    dir = str(pathlib.Path.home())
    native_file = os.path.join(dir, filename)
    wsl_file = None

    if sys.platform == 'win32':
        breakdown = native_file.replace(':', '').split(sep='\\')
        breakdown[0] = breakdown[0].lower()
        breakdown = ['mnt'] + breakdown
        wsl_file = '/' + '/'.join(breakdown)
    return native_file, wsl_file

def wsl2win(path):
    """Convert WSL path to Windows path

    Parameters
    ----------
    path: String
        Path in WSL format (/mnt/d/....)
    Returns
    -------
    win_path: String
        Path in Windows format

    """
    win_path = [wp for wp in path.replace('/mnt/', '')]
    win_path.insert(win_path.index('/'), ':\\')
    win_path.pop(win_path.index('/'))
    win_path = ''.join(win_path).replace('/','\\')
    return win_path


def win2wsl(path):
    """Convert WIndows path tp WSL path

    Parameters
    ----------
    path: String
        Path in Windows format
    Returns
    -------
    win_path: String
        Path in WSL format (/mnt/d/....)

    """
    wsl_path = '//mnt//' + path.replace(':\\','//').replace('\\','//')
    return wsl_path


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
