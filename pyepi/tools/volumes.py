"""
Volume manipulation toolkit.
"""

import numpy as np
import nibabel as nib
import pandas as pd


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


def identify_voxel_location(coords, atlas_volume, lut_table):
    """ Identify structure of each voxel, based on segmentation in atlas_volume.

    coords: Pandas dataframe
        Contains at least the following columns: name, xmrivox, ymrivox, zmrivox
    atlas_volume: string
        Path to atlas segmentation volume (Freesurfer style, ex: aparg+aseg.mgz)
    lut_table: string
        Path to LUT table (Freesurfer Style) in xlsx format

    Returns:

    coords: Pandas dataframe
        Update coords dataframe with two additional column, showing the exact brain structure of each voxel,
        and the brain structure from which it is most likely to record brain activity from

    """
    atlas = nib.load(atlas_volume).get_data()
    lut = pd.read_excel(lut_table)

    wm_and_unknown = lut[lut['Name'].str.contains('Unknown')]['No'].values
    wm_and_unknown = np.append(wm_and_unknown, lut[lut['Name'].str.contains('Matter')]['No'].values)

    center_voxel = []
    most_likely = []
    for i in np.arange(0, coords.shape[0]):
        x = np.round(coords.loc[i]['xmrivox']).astype(np.int)
        y = np.round(coords.loc[i]['ymrivox']).astype(np.int)
        z = np.round(coords.loc[i]['zmrivox']).astype(np.int)
        atlas_value = atlas[x][y][z]
        center_voxel.append(lut[lut['No'] == atlas_value]['Name'].values[0])
        v = np.zeros((3, 3, 3))

        # search the center voxel and the 8 connected voxels.
        v[0][0][0] = atlas[x - 1][y - 1][z - 1]
        v[0][0][1] = atlas[x - 1][y - 1][z]
        v[0][0][2] = atlas[x - 1][y - 1][z + 1]

        v[0][1][0] = atlas[x - 1][y][z - 1]
        v[0][1][1] = atlas[x - 1][y][z]
        v[0][1][2] = atlas[x - 1][y][z + 1]

        v[0][2][0] = atlas[x - 1][y + 1][z - 1]
        v[0][2][1] = atlas[x - 1][y + 1][z]
        v[0][2][2] = atlas[x - 1][y + 1][z + 1]

        v[1][0][0] = atlas[x][y - 1][z - 1]
        v[1][0][1] = atlas[x][y - 1][z]
        v[1][0][2] = atlas[x][y - 1][z + 1]

        v[1][1][0] = atlas[x][y][z - 1]
        v[1][1][1] = atlas[x][y][z]
        v[1][1][2] = atlas[x][y][z + 1]

        v[1][2][0] = atlas[x][y + 1][z - 1]
        v[1][2][1] = atlas[x][y + 1][z]
        v[1][2][2] = atlas[x][y + 1][z + 1]

        v[2][0][0] = atlas[x + 1][y - 1][z - 1]
        v[2][0][1] = atlas[x + 1][y - 1][z]
        v[2][0][2] = atlas[x + 1][y - 1][z + 1]

        v[2][1][0] = atlas[x + 1][y][z - 1]
        v[2][1][1] = atlas[x + 1][y][z]
        v[2][1][2] = atlas[x + 1][y][z + 1]

        v[2][2][0] = atlas[x + 1][y + 1][z - 1]
        v[2][2][1] = atlas[x + 1][y + 1][z]
        v[2][2][2] = atlas[x + 1][y + 1][z + 1]

        histo = pd.Series(v.ravel()).value_counts().to_frame().sort_values(by=0, axis=0,
                                                                           ascending=False)
        histo_clean = histo.loc[[i for i in histo.index if i not in wm_and_unknown]].sort_values(by=0, axis=0,
                                                                                                 ascending=False)
        if histo_clean.shape[0] > 0:
            most_likely.append(lut[lut['No'] == histo_clean.index[0].astype(int)]['Name'].values[0])
        else:
            most_likely.append(lut[lut['No'] == histo.index[0].astype(int)]['Name'].values[0])
    coords = pd.concat([coords,
                        pd.DataFrame(center_voxel, columns=['exact']),
                        pd.DataFrame(most_likely, columns=['most_likely']),
                        ], axis=1)
    return coords
