"""
Volume manipulation toolkit.
"""

import numpy as np
import nibabel as nib
import os


def normalize(input_volume, output_volume, normalize_by='max'):
    """ Normalize voxels in a volume

    Parameters
    ----------
    input_volume: string
        Path to input volume.
    output_volume: string
        Path to output volume
    normalize_by: string / float / int
        If float, the volume will be normalized to the float value.
        If "max", the volume will be normalized to the maximum voxel volume in the input volume
    """
    done = False
    vol = nib.load(input_volume)
    data = vol.get_data()
    if isinstance(normalize_by, str) and normalize_by is 'max':
        data = np.divide(data, np.max(data))
        done = True
    if (isinstance(normalize_by, float)) or (isinstance(normalize_by, int)):
        data = np.divide(data, normalize_by)
        done = True
    if done:
        new_vol = nib.Nifti1Image(data, vol.affine)
        nib.save(new_vol, output_volume)
