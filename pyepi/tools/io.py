"""
Collection of functions for reading/writing  various file formats.

TODO: ppr

"""

import numpy as np
import pandas as pd
import os


def read_ppr(filename):
    """
    Read .ppr files (Waypoint Planner/Navigator)

    Parameters
    ----------
    filename: string
        Path to .ppr file

    Returns
    -------
    ppr_scans: dict
        Dictionary with information about images. Xfm field is the transformation matrix from each image to the
        reference image (which may or may not be I1)
    ppr_anat: dict
        Coordinates of AC, PC and midplane points in patient coordinates
    ppr_trajectories: dict of dicts
        Each dictionary containes trajectory name, entry and target points in patient coordinates.

    """

    uids = {}
    xfms = {}
    ppr_scans = {}
    ppr_anat = {}
    ppr_trajectories = {}
    ppr = tuple(open(filename, 'r'))
    n = 0
    while n < len(ppr):
        if ('Series UID' in ppr[n]) & ('//' not in ppr[n]):
            sstr = ppr[n].replace('\n', '').split(sep=' ')
            uids[sstr[0]] = sstr[-1]
        if ('IMAGE INFO' in ppr[n]) & ('//' not in ppr[n]) & ('REF' not in ppr[n]):
            sstr = ppr[n].replace('\n', '').split(sep=' ')
            ppr_scans[sstr[0]] = {}
            ppr_scans[sstr[0]]['type'] = sstr[17]
            ppr_scans[sstr[0]]['size'] = np.array(sstr[5:8], dtype=int)
            ppr_scans[sstr[0]]['pixel_size'] = np.array(sstr[8:11], dtype=float)
            ppr_scans[sstr[0]]['orientation'] = np.array(sstr[11:14], dtype=int)
            # compute world transforms
            if np.array_equal(ppr_scans[sstr[0]]['orientation'], np.array([1, 2, 3])):
                # axial
                ppr_scans[sstr[0]]['wt'] = np.array(
                    [[-1, 0, 0, (ppr_scans[sstr[0]]['size'][0] - 1) * ppr_scans[sstr[0]]['pixel_size'][0] / 2],
                     [0, 1, 0, -(ppr_scans[sstr[0]]['size'][1] - 1) * ppr_scans[sstr[0]]['pixel_size'][1] / 2],
                     [0, 0, -1, (ppr_scans[sstr[0]]['size'][2] - 1) * ppr_scans[sstr[0]]['pixel_size'][2] / 2],
                     [0, 0, 0, 1]])
            if np.array_equal(ppr_scans[sstr[0]]['orientation'], np.array([2, 3, 1])):
                # saggital
                ppr_scans[sstr[0]]['wt'] = np.array(
                    [[-1, 0, 0, (ppr_scans[sstr[0]]['size'][2] - 1) * ppr_scans[sstr[0]]['pixel_size'][2] / 2],
                     [0, 1, 0, -(ppr_scans[sstr[0]]['size'][0] - 1) * ppr_scans[sstr[0]]['pixel_size'][0] / 2],
                     [0, 0, -1, (ppr_scans[sstr[0]]['size'][1] - 1) * ppr_scans[sstr[0]]['pixel_size'][1] / 2],
                     [0, 0, 0, 1]])
            if np.array_equal(ppr_scans[sstr[0]]['orientation'], np.array([1, -3, 2])):
                # coronal
                ppr_scans[sstr[0]]['wt'] = np.array(
                    [[-1, 0, 0, (ppr_scans[sstr[0]]['size'][0] - 1) * ppr_scans[sstr[0]]['pixel_size'][0] / 2],
                     [0, 1, 0, -(ppr_scans[sstr[0]]['size'][2] - 1) * ppr_scans[sstr[0]]['pixel_size'][2] / 2],
                     [0, 0, -1, (ppr_scans[sstr[0]]['size'][1] - 1) * ppr_scans[sstr[0]]['pixel_size'][1] / 2],
                     [0, 0, 0, 1]])

        if ('ALIAS' in ppr[n]):
            sstr = ppr[n].replace('\n', '').split(sep=' ')
            ppr_scans[sstr[0]]['alias'] = sstr[9]
        if ('[TRANSFORMATION ' in ppr[n]):
            sstr = ppr[n].replace('\n', '').split(sep=' ')
            key = sstr[1].replace('->REF]', '')
            sstr = [ppr[n + 1], ppr[n + 2], ppr[n + 3], ppr[n + 4]]
            xfms[key] = np.array([float(i) for i in ''.join(sstr)
                                 .replace('\\n', '').split()]).reshape((4, 4))
        if ('ANATOMY' in ppr[n]):
            ppr_anat['AC'] = np.array([float(i) for i in ppr[n + 1].replace('\n', '').split(sep=' ') if len(i)],
                                      dtype=float)
            ppr_anat['PC'] = np.array([float(i) for i in ppr[n + 2].replace('\n', '').split(sep=' ') if len(i)],
                                      dtype=float)
            ppr_anat['MP'] = np.array([float(i) for i in ppr[n + 3].replace('\n', '').split(sep=' ') if len(i)],
                                      dtype=float)
        if ('TRAJECTORY' in ppr[n]):
            n += 1
            while '"' in ppr[n]:
                sstr = [i for i in ppr[n].replace('\n', '').replace('"', '').split(sep=' ') if len(i)]
                ppr_trajectories[sstr[-1]] = {}
                ppr_trajectories[sstr[-1]]['tp'] = np.array(sstr[0:3], dtype=float)
                ppr_trajectories[sstr[-1]]['ep'] = np.array(sstr[3:6], dtype=float)
                n += 1
        n += 1
    for k in ppr_scans.keys():
        ppr_scans[k]['uid'] = uids[k]
        ppr_scans[k]['xfm'] = xfms[k]
    return ppr_scans, ppr_anat, ppr_trajectories


def load_contacts(filename):
    """
    Loads contact coordinates from a file.

    File is an Excel file with a sheet named "ContactCoordinates", which has 4 columns: Name,	X,	Y,	Z.
    Name is the contact name, following the French notation where the apostrophe marks left hemispheric contacts (ex:
    right amygdala contact is A01, left amygdala contact is A'01).
    The first row is the header, the next 3 rows are
    AC, PC and MP (anterior, posterior comissure, and midplane points), the 5th row contains info about the MRI/CT space
    in which the coordinates are expressed (series UID, then volume dimensions). Contacts start on the 6th row.

    Parameters
    ----------
    filename: string
        Path to file containing contacts coordinates.
    Returns
    -------

    """
    _, file_extension = os.path.splitext(filename)
    if file_extension not in ['.xls', '.xlsx']:
        print('File type not supported.')
        return
    xls = pd.ExcelFile(filename)
    data = xls.parse(sheet_name='ContactCoordinates')
    landmarks = data.loc[0:2, :]
    mri_uid = data.values[3, 0]
    coords = data.loc[4:, ['Name', 'X', 'Y', 'Z']].reset_index(drop=True)
    coords = coords.assign(hemi=pd.Series(np.random.randn(coords.shape[0])).values)
    coords.loc[coords['Name'].str.contains("'") == True, 'hemi'] = 'L'
    coords.loc[coords['Name'].str.contains("'") == False, 'hemi'] = 'R'
    coords.columns = [c.lower() for c in coords.columns]  # lower case
    landmarks.columns = [c.lower() for c in landmarks.columns]  # lower case
    return coords, landmarks, mri_uid


def read_afni_shift(filename):
    """Read AFNI _shft.1D format
    Which is really a 1 line text files with with a 3x4 transformation matrix. The last line needed for a 4x4 matrix is
    added automatically/

    Parameters
    ----------
    filename: string
        Path to a AFNI/SUMA generated _shft.1D file.

    Returns
    -------
    xfm: ndarray
        4x4 transformation matrix
    """
    with open(filename) as f:
        xfm = f.read()
    xfm = xfm.split()
    xfm = np.array(xfm, dtype=float)
    xfm = np.append(xfm, [0, 0, 0, 1])
    xfm = xfm.reshape((4, 4))
    return xfm
