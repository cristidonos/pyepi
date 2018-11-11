"""
Collection of plotting functions.

"""

from pyepi.tools import inout, viz
import os
import numpy as np
import pandas as pd
from mayavi import mlab
from surfer import Brain
from scipy.stats import zscore
import matplotlib.pyplot as plt

"""
GENERAL PURPOSE PLOTTING FUNCTIONS -- MOVE TO NEW MODULE/PACKAGE?
"""


def bernstein(n, j, t):
    """ Compute Bernstein polynomials
    """
    factorial = np.math.factorial
    b = factorial(n) / (factorial(j) * factorial(n - j)) * np.power(t, j) * np.power(1 - t, n - j)
    return b


def bezier3d(points, npoints=100):
    """ Bezier interpolation for given points.
    """
    Q = np.zeros((3, npoints))
    t = np.linspace(0, 1, npoints)
    for k in np.arange(0, len(t)):
        for j in np.arange(0, points.shape[1]):
            Q[:, k] = Q[:, k] + points[:, j] * bernstein(points.shape[1] - 1, j, t[k])
    return Q


"""
END OF GENERAL PURPOSE PLOTTING FUNCTIONS
"""


def electrode_labels(coords, brain, electrode_label_size=2):
    """ Plot electrode labels to SEEG electrodes.

    Parameters
    ----------
    coords: Pandas dataframe
        Dataframe containinf contact coordinates
    brain: pysurfer handle
        Brain handle generated from pysurfer (viz.Brain)
    electrode_label_size: float
        Size of text label

    Returns
    -------
    brain: pysurfer handle
        Brain handle with added text labels

    """
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


def get_views(coords):
    """
    Decide best SAG, AX and COR views based on electrode locations

    Parameters
    ----------
    coords: Pandas dataframe
        Dataframe of contacts coordinates.

    Returns
    -------
    views: list
        List of suggested views.

    """
    views = []
    if coords['xmri'].mean() > 0:
        views.append('ros')
    else:
        views.append('cau')
    if coords['ymri'].mean() > 0:
        views.append('lat')
    else:
        views.append('med')
    if coords['zmri'].mean() > 0:
        views.append('dor')
    else:
        views.append('ven')
    #views.append({'azimuth':179, 'elevation':89})  # dorsal view does not work properly. use ventral for now. TODO: check dorsal plots

    return views


def implantation_scheme(subj, SUBJECTS_DIR_NATIVE, fig_size=(1000, 800), electrode_label_size=2, brain_alpha=0.3):
    coords = pd.read_excel(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))
    brain = Brain(subj, 'both', 'pial', subjects_dir=SUBJECTS_DIR_NATIVE, alpha=brain_alpha, offset=False,
                  cortex='low_contrast', background='white', size=fig_size, views=['lat'],
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


def spes_responses_by_contacts(subj, SUBJECTS_DIR_NATIVE, contacts=None, flow='outbound', plot_type='circle',
                               filter_spearman_r=0.5,
                               filter_spearman_p=0.05, filter_zscore=3, percentile_threshold = 75,
                               fig_size=(1000, 800), electrode_label_size=2,
                               brain_alpha=0.3, use_bezier=True,
                               visible=True, max_linewidth=3, hide_buttons=False):
    """ Plot SPES / CCEP responses on 3d brain or on a circular graph

    Parameters
    ----------
    subj: string
        Subject ID
    SUBJECTS_DIR_NATIVE: path
        Path to Freesurfer's SUBJECTS_DIR in native OS format
    contacts: list
        List of contact names. If "plot_type" is brain, only connections from contacts in this list will be drawn. If
        "plot_type" is circle, contacts in this list will be shown by default in the circular plot.
    flow: string
        Can be "inbound" or "outbound".
    plot_type: string
        Can be "brain" or "circle"
    filter_spearman_r: float
        Threshold Spearman'r for SPES responses
    filter_spearman_p: float
        Threshold Spearman'p for SPES responses
    filter_zscore: float
        Threshold zscore for SPES responses. This is helpful to remove outliers, for eaxmple contacts nearby the
        stimulation site exhibiting stimulation artefacts.
    percentile_threshold: float
        Percentile threshold. Will keep responses higher than this threshold. Default Q3 (75%)
    fig_size: tuple
        Figure size in (x,y) format
    electrode_label_size: float
        Size of text label when plotting on the brain. Ignored for circular plots.
    brain_alpha: float
        Alpha for brain surface. Ignored for circular plots/
    use_bezier: bool
        If True will use bezier curves instead of lines to plot connections on the 3d brain. Ignored for circular plots.
    visible: bool
        If True, all connections will be shown when the circular plot is first drawn. Ignored for 3d brain plot
    max_linewidth: float
        Maximum linewidth used to normalize connections in circular plot. Ignored for 3d brain plot.
    hide_buttons:  bool
        If True, Show all and Hide all buttons will not be displayed in circular plot.

    Returns
    -------
    brain: pysurfer handle
        Handle to pysurfer's brain with added connections, electrodes, labels, etc. Empty for circular plots
    circle: CircularGraph handle
        Handle to circular graph object. Empty for 3d brain plots.
    spes: Pandas dataframe
        Dataframe with SPES responses, thresholded according to input arguments
    """
    brain = None
    circle = None
    spes, sheetname = inout.load_spes(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'SPES', 'SPES.xls'), sheetname='SEEG')
    coords = pd.read_excel(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))
    all_contacts = set(spes['RespContact'].values)

    if filter_zscore is not None:
        # filter out outliers > XX SD (mostly channels adjacent to stimulation site, but who knows what other artefacts...
        spes = spes[zscore(spes['mean_rms']) < filter_zscore].reset_index(drop=True)

    if  percentile_threshold is not None:
        q3 = np.percentile(spes['mean_rms'].values, percentile_threshold)
        spes = spes[spes['mean_rms'] > q3].reset_index(drop=True)
    else:
        q3 = 0


    if (filter_spearman_r is not None) and (filter_spearman_p is not None):
        spes = spes[(spes['pvalue'] < filter_spearman_p) & (spes['correlation'] > filter_spearman_r)].reset_index(
            drop=True)

    # filter out responses on stimulation contacts
    spes = spes[spes.apply(lambda x: x['RespContact'] not in x['StimContact'], axis=1)].reset_index(drop=True)


    if plot_type == 'brain':
        # Plot Brain and contacts
        brain = Brain(subj, 'both', 'pial', subjects_dir=SUBJECTS_DIR_NATIVE, alpha=brain_alpha, offset=False,
                      cortex='low_contrast', background='black', size=fig_size, views=['lat'],
                      title=subj + ' - ' + flow)
        brain.add_foci(coords.loc[coords['hemi'] == 'L', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                       hemi='lh', color='white')
        brain.add_foci(coords.loc[coords['hemi'] == 'R', ['xmri', 'ymri', 'zmri']].values, scale_factor=0.2,
                       hemi='rh', color='white')

    stim_pairs = set(spes['StimContact'])

    if plot_type == 'brain':
        # keep only contacts of interest, defined my contacts and flow arguments
        if contacts is not None:
            if flow == 'outbound':
                spes = spes[spes['StimContact'].str.contains('|'.join(contacts))].reset_index(drop=True)
            if flow == 'inbound':
                spes = spes[spes['RespContact'].str.contains('|'.join(contacts))].reset_index(drop=True)

        lh_contacts = list(coords[coords['hemi'] == 'L']['name'].values)
        rh_contacts = list(coords[coords['hemi'] == 'R']['name'].values)
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
                mlab.plot3d(x, y, z, color=(1, 0.5, 0.5), tube_radius=0.5)
                data = spes[spes['StimContact'] == sp]
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

    if plot_type == 'circle':
        # norm responses to 1
        spes['mean_rms'] = spes['mean_rms'] / max(spes['mean_rms'])
        # redundant.. but make sure stim pairs are included.
        stim_contacts = []
        for sp in stim_pairs:
            alphanum = [int(c.isnumeric()) for c in sp]
            middle = alphanum.index(1) + alphanum[alphanum.index(1):].index(0)
            stim_contacts.append(sp[:middle])

        all_contacts = list(all_contacts.union(set(stim_contacts)))
        all_contacts.sort()
        contact_colors = plt.cm.rainbow(np.linspace(0, 1, len(all_contacts)))

        # figure oput electrodes and assign a color for each one
        all_electrodes = list(set([ac[:np.where([d.isdigit() for d in ac])[0][0]] for ac in all_contacts]))
        all_electrodes.sort()
        electrode_colors = plt.cm.rainbow(np.linspace(0, 1, len(all_electrodes)))

        # create adjacency matrix
        adjacency_matrix = np.zeros((len(all_contacts), len(all_contacts)))
        for sp in stim_pairs:
            try:
                alphanum = [int(c.isnumeric()) for c in sp]
                middle = alphanum.index(1) + alphanum[alphanum.index(1):].index(0)
                contact1 = sp[:middle]
                ix = all_contacts.index(contact1)
                if flow == 'outbound':
                    resps = spes[spes['StimContact'] == sp].reset_index(drop=True)
                    for r in np.arange(0, resps.shape[0]):
                        adjacency_matrix[ix][all_contacts.index(resps.iloc[r]['RespContact'])] = resps.iloc[r][
                            'mean_rms']
                if flow == 'inbound':
                    resps = spes[spes['StimContact'] == sp].reset_index(drop=True)
                    for r in np.arange(0, resps.shape[0]):
                        adjacency_matrix[all_contacts.index(resps.iloc[r]['RespContact'])][ix] = resps.iloc[r][
                            'mean_rms']

            except:
                print(sp)

        circle = viz.CircularGraph(adjacency_matrix, all_contacts, connection_colors=contact_colors,
                                   node_colors=contact_colors, visible=visible, highlighted_labels=contacts,
                                   max_linewidth=max_linewidth, hide_buttons = hide_buttons)
        circle.ax.annotate(', '.join([subj, flow]),
                           xy=(0, 0), xycoords='axes fraction',
                           xytext=(-20, -20), textcoords='offset pixels',
                           horizontalalignment='left',
                           verticalalignment='bottom')

    return brain, circle, spes

