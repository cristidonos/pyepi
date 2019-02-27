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


def identify_voxel_location(coords, atlas_volume, lut_table, name_prefix=None):
    """ Identify structure of each voxel, based on segmentation in atlas_volume.

    coords: Pandas dataframe
        Contains at least the following columns: name, xmrivox, ymrivox, zmrivox
    atlas_volume: string
        Path to atlas segmentation volume (Freesurfer style, ex: aparg+aseg.mgz)
    lut_table: string
        Path to LUT table (Freesurfer Style) in xlsx format
    name_prefix: string
        Some prefix to be added to the "exact" and "most_likely" columns

    Returns:

    coords: Pandas dataframe
        Update coords dataframe with two additional column, showing the exact brain structure of each voxel,
        and the brain structure from which it is most likely to record brain activity from

    """
    atlas = nib.load(atlas_volume).get_data()
    if len(atlas.shape)>3:
        atlas = np.squeeze(atlas)

    lut = pd.read_excel(lut_table)

    wm_and_unknown = lut[lut['Name'].str.contains('Unknown', case=False)]['No'].values
    wm_and_unknown = np.append(wm_and_unknown, lut[lut['Name'].str.contains('Matter', case=False)]['No'].values)

    center_voxel = []
    most_likely = []
    for i in np.arange(0, coords.shape[0]):
        if name_prefix is None:
            x = np.round(coords.loc[i]['xmrivox']).astype(np.int)
            y = np.round(coords.loc[i]['ymrivox']).astype(np.int)
            z = np.round(coords.loc[i]['zmrivox']).astype(np.int)
        else:
            x = np.round(coords.loc[i][name_prefix+ '_xmrivox']).astype(np.int)
            y = np.round(coords.loc[i][name_prefix+ '_ymrivox']).astype(np.int)
            z = np.round(coords.loc[i][name_prefix+ '_zmrivox']).astype(np.int)

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
    if name_prefix is None:
        ecol = 'exact'
        mlcol = 'most_likely'
    else:
        ecol = name_prefix + '_' + 'exact'
        mlcol = name_prefix + '_' + 'most_likely'

    coords = pd.concat([coords,
                        pd.DataFrame(center_voxel, columns=[ecol]),
                        pd.DataFrame(most_likely, columns=[mlcol]),
                        ], axis=1)
    return coords


def average_structure_coordinates(atlas_volume, lut_table):
    """ Computes the mean voxel coordinates of each brain structure in atlas_volume.

    atlas_volume: string
        Path to atlas segmentation volume (Freesurfer style, ex: aparg+aseg.mgz)
    lut_table: string
        Path to LUT table (Freesurfer Style) in xlsx format

    Returns:

    coords: Pandas dataframe
        X, Y, Z coordinates for each brain structure in scan and voxel coordinates

    """

    atlas_vol = nib.load(atlas_volume)
    atlas = atlas_vol.get_data()
    vox2ras = atlas_vol.header.get_vox2ras_tkr()
    lut = pd.read_excel(lut_table)

    wm_and_unknown = lut[lut['Name'].str.contains('Unknown', case=False)]['No'].values
    wm_and_unknown = set(np.append(wm_and_unknown, lut[lut['Name'].str.contains('Matter', case=False)]['No'].values))

    coords = pd.DataFrame(columns=['name', 'hemi', 'xmri', 'ymri', 'zmri', 'xmrivox', 'ymrivox', 'zmrivox'])
    labels = set(np.unique(atlas)) - wm_and_unknown
    for label in labels:
        xmrivox, ymrivox, zmrivox = np.round(np.mean(np.stack(np.where(atlas == label)).T, axis=0)).astype(np.int)
        xmri, ymri, zmri, _ = np.dot(vox2ras, np.array([xmrivox, ymrivox, zmrivox, 1]))
        name = lut[lut['No'] == label]['Name'].values[0]
        hemi = ''
        if any(x in name for x in ['-rh-', 'Right']):
            hemi = 'R'
        if any(x in name for x in ['-lh-', 'Left']):
            hemi = 'L'
        data = {'name': [name], 'hemi': [hemi], 'xmri': [xmri], 'ymri': [ymri], 'zmri': [zmri], 'xmrivox': [xmrivox],
                'ymrivox': [ymrivox], 'zmrivox': [zmrivox]}
        coords = pd.concat([coords,
                            pd.DataFrame(data=data),
                            ], axis=0)
    coords = coords.sort_values(by=['hemi', 'name']).reset_index(drop=True)

    return coords


def contact_to_volume(contact_coords, reference_volume, contact_volume):
    """ Takes the reference volume and create a one-voxel volume in the same space, containing the contact coordinate.

    Parameters
    ----------
    contact_coords: np array or list
        Contact coordinates in voxels in reference_volume's space
    reference_volume: string
        Path to reference volume
    contact_volume: string
        Path to use for saving output volume

    Returns
    -------

    """
    vol = nib.load(reference_volume)
    data = vol.get_data() * 0
    data[contact_coords[0]][contact_coords[1]][contact_coords[2]] = 100
    new_vol = nib.Nifti1Image(data, vol.affine)
    nib.save(new_vol, contact_volume)


def contact_volume_to_mni_coordinates(contact_volume):
    """ Takes a contact volume and return contact coordinates in voxels and MNI coordinates

    Parameters
    ----------
    contact_volume: string
        Path to contact volume

    Returns
    -------
    mri_coords: numpy array
        Intensity weighted average of x, y, z in MRI coordinates
    mri_vox: numpy array
        Intensity weighted average of x, y, z in MRI voxel coordinates (rounded)
    mri_stats: dictionary
        Dictionary of numbers of non-zero voxels, max voxel intensity , mean and std deviation of voxel intensity

    """
    vol = nib.load(contact_volume)
    data = vol.get_data()

    voxels = np.where(data > 0)
    nvoxels = voxels[0].shape[0]  # number of non-zero voxels
    max_voxel_intensity = np.max(data[voxels])
    mean_voxel_intensity = np.mean(data[voxels])
    std_voxel_intensity = np.std(data[voxels])
    xmrivox, ymrivox, zmrivox = [np.average(voxels[0], weights=data[voxels]),
                                 np.average(voxels[1], weights=data[voxels]),
                                 np.average(voxels[2], weights=data[voxels])]
    # RAS coordinates in MNI
    xmri, ymri, zmri, _ = np.round(np.dot(np.array([xmrivox, ymrivox, zmrivox, 1]), vol.header.get_vox2ras().T),
                                   decimals=3)
    # voxel coordinates in MNI
    xmrivox, ymrivox, zmrivox = np.int16(np.round(np.array([xmrivox, ymrivox, zmrivox])))

    mri_coords = np.array([xmri, ymri, zmri])
    mri_vox = np.array([xmrivox, ymrivox, zmrivox])
    mri_stats = {'nvoxels': nvoxels,
                 'max_voxel_intensity' : max_voxel_intensity,
                 'mean_voxel_intensity': mean_voxel_intensity,
                 'std_voxel_intensity' : std_voxel_intensity}
    return mri_coords, mri_vox, mri_stats
