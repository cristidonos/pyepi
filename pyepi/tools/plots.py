"""
Collection of plotting functions.

"""

from pyepi.tools import io, paths
import os
import numpy as np
import pandas as pd
from mayavi import mlab
from surfer import Brain
from scipy.stats import zscore


"""
GENERAL PURPOSE PLOTTING FUNCTIONS -- MOVE TO NEW MODULE/PACKAGE?
"""

def bernstein(n,j,t):
    """ Compute Bernstein polynomials
    """
    factorial = np.math.factorial
    b = factorial(n) / ( factorial(j) * factorial(n-j) ) * np.power(t,j) * np.power(1-t,n-j)
    return b

def bezier3d(points, npoints=100):
    """ Bezier interpolation for given points.
    """
    Q = np.zeros((3,npoints))
    t = np.linspace(0,1, npoints)
    for k in np.arange(0,len(t)):
        for j in np.arange(0,points.shape[1]):
            Q[:,k]=Q[:,k] + points[:,j] * bernstein(points.shape[1] -1, j, t[k])
    return Q

"""
END OF GENERAL PURPOSE PLOTTING FUNCTIONS
"""


def implantation_scheme(subj, SUBJECTS_DIR_NATIVE, electrode_label_size=2, brain_alpha=0.3):
    coords = pd.read_excel(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))
    brain = Brain(subj, 'both', 'pial', subjects_dir=SUBJECTS_DIR_NATIVE, alpha=brain_alpha, offset=False,
                  cortex='low_contrast', background='black', size=(1000, 800), views=['lat'],
                  title=subj + ' - implantation scheme')
    brain.add_foci(coords.loc[coords['hemi'] == 'L', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                   hemi='lh', color='red')
    brain.add_foci(coords.loc[coords['hemi'] == 'R', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                   hemi='rh', color='red')

    # find outer contact and prin electrode labels
    outer_contacts = []
    for k in np.arange(1, coords.shape[0]):
        if coords.loc[k - 1]['name'][0] != coords.loc[k]['name'][0]:
            outer_contacts.append(coords.loc[k - 1]['name'])
    outer_contacts.append(coords.iloc[-1]['name'])  # last contact added by default

    for c in outer_contacts:
        row = coords[coords['name'] == c]
        electrode_name = []
        electrode_name.append([c for c in row['name'].values[0] if not c.isdigit()])
        electrode_name = ''.join(electrode_name[0])

        brain.add_text3d(row['xmri'].values[0] + 2, row['ymri'].values[0] + 2, row['zmri'].values[0] + 2,
                         text=electrode_name, name=electrode_name)
        brain.texts_dict[electrode_name]['text'].scale = np.array([electrode_label_size,
                                                                   electrode_label_size, electrode_label_size])
    return brain


def spes_responses(subj, SUBJECTS_DIR_NATIVE, modality='SEEG', group_by_structures=False, filter_spearman_r=0.5,
                   filter_spearman_p=0.05, filter_zscore=3, electrode_label_size=2, brain_alpha=0.3, use_bezier=True):
    spes, sheetname = io.load_spes(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'SPES', 'SPES.xls'), sheetname=modality)
    coords = pd.read_excel(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))
    lh_contacts = list(coords[coords['hemi'] == 'L']['name'].values)
    rh_contacts = list(coords[coords['hemi'] == 'R']['name'].values)

    if (filter_spearman_r is not None) and (filter_spearman_p is not None):
        spes = spes[(spes['pvalue'] < filter_spearman_p) & (spes['correlation'] > filter_spearman_r)].reset_index(
            drop=True)

    # filter out responses on stimulation contacts
    spes = spes[spes.apply(lambda x: x['RespContact'] not in x['StimContact'], axis=1)].reset_index(drop=True)

    if filter_zscore is not None:
        # filter out outliers > XX SD (mostly channels adjacent to stimulation site, but who knows what other artefacts...
        spes = spes[zscore(spes['mean_rms']) < filter_zscore].reset_index(drop=True)

    # norm responses to 1
    spes['mean_rms'] = spes['mean_rms'] / max(spes['mean_rms'])
    # spes['mean_rms'] = spes['mean_rms'] / spes['mean_rms'].median()

    # Plot Brain and contacts
    brain = Brain(subj, 'both', 'pial', subjects_dir=SUBJECTS_DIR_NATIVE, alpha=brain_alpha, offset=False,
                  cortex='low_contrast', background='black', size=(1000, 800), views=['lat'],
                  title=subj + ' - SPES responses')
    brain.add_foci(coords.loc[coords['hemi'] == 'L', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                   hemi='lh', color='white')
    brain.add_foci(coords.loc[coords['hemi'] == 'R', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                   hemi='rh', color='white')

    stim_pairs = set(spes['StimContact'])

    for sp in stim_pairs:
        try:
            alphanum = [int(c.isnumeric()) for c in sp]
            middle = alphanum.index(1) + alphanum[alphanum.index(1):].index(0)
            contact1 = sp[:middle]
            contact2 = sp[middle:]
            contact1_coords = coords[coords['name'] == contact1][['xmri', 'ymri', 'zmri']].values[0]
            contact2_coords = coords[coords['name'] == contact2][['xmri', 'ymri', 'zmri']].values[0]
            midpoint_coords = np.mean(np.vstack([contact1_coords, contact2_coords]), axis=0)
            x, y, z = zip(contact1_coords, contact2_coords)
            mlab.plot3d(x, y, z, color=(1,0.5,0.5), tube_radius=0.5)
            data = spes[spes['StimContact']==sp]
            if use_bezier:
                tb1 = 0.35  # Tightness factor for trunk
                tb2 = 0.05  # Tightness factor for branches
                cm_offset = 2
                lh_resp_contacts = set(data['RespContact'].values).intersection(lh_contacts)
                rh_resp_contacts = set(data['RespContact'].values).intersection(rh_contacts)

                if len(lh_contacts) > 0:
                    CM_lh = coords.iloc[np.where(coords['name'].isin(lh_resp_contacts))][
                        ['xmri', 'ymri', 'zmri']].mean().values
                    C0_lh = midpoint_coords + tb1 * (CM_lh - midpoint_coords) + cm_offset * (
                        CM_lh - midpoint_coords) / np.linalg.norm(CM_lh - midpoint_coords)
                if len(rh_contacts) > 0:
                    CM_rh = coords.iloc[np.where(coords['name'].isin(rh_resp_contacts))][
                        ['xmri', 'ymri', 'zmri']].mean().values
                    C0_rh = midpoint_coords + tb1 * (CM_rh - midpoint_coords) + cm_offset * (
                        CM_rh - midpoint_coords) / np.linalg.norm(CM_rh - midpoint_coords)

            for rc in set(data['RespContact']):
                rc_coords = coords[coords['name'] == rc][['xmri', 'ymri', 'zmri']].values[0]
                rms_value = data[data['RespContact'] == rc]['mean_rms'].values[0]
                if use_bezier:
                    if coords[coords['name'] == rc]['hemi'].values == 'L':
                        CP = rc_coords + tb2 * (midpoint_coords - CM_lh)
                        b = bezier3d(np.stack([midpoint_coords, C0_lh, CP, rc_coords]).T, 100).T
                    if coords[coords['name'] == rc]['hemi'].values == 'R':
                        CP = rc_coords + tb2 * (midpoint_coords - CM_rh)
                        b = bezier3d(np.stack([midpoint_coords, C0_rh, CP, rc_coords]).T, 100).T
                    mlab.plot3d(b[:, 0], b[:, 1], b[:, 2], color=(0.5, 1, 0.5), tube_radius=0.75 * rms_value)
                else:
                    x, y, z = zip(midpoint_coords, rc_coords)
                    mlab.plot3d(x, y, z, color=(0.5, 1, 0.5), tube_radius=0.75 * rms_value)
        except:
            print('Failed to load and plot ' + sp)

# from pyepi.tools import plots
# plots.spes_responses('SEEG72',r'D:\CloudSynology\subjects')
