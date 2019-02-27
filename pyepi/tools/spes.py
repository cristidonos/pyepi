"""
Functions for processing CCEPs from single pulse electrical stimulation (SPES) as implemented at University of Bucharest

"""

import pandas as pd
import numpy as np
import os
from pyepi.tools import paths, inout, spes
from scipy.stats import zscore
import itertools


def get_stim_contact(table):
    stim_pairs = list(table['StimContact'].unique())
    contact1 = []
    contact2 = []
    for sp in stim_pairs:
        alphanum = [int(c.isnumeric()) for c in sp]
        middle = alphanum.index(1) + alphanum[alphanum.index(1):].index(0)
        contact1.append(sp[:middle])
        contact2.append(sp[middle:])
    return stim_pairs, contact1, contact2


def get_cceps(subj, protocol='SPES', lowfreq=0, highfreq=0, rmswindow=100, rmswindowstart=10, filter_spearman_r=0.5,
              filter_spearman_p=0.05,
              filter_zscore=3, percentile_threshold=75):
    """

    Parameters
    ----------
    subj
    protocol: string
        Protocol column in SPES data file
    lowfreq: float
        LowFreq value in SPES data file
    highfreq: float
        HighFreq value in SPES data file
    rmswindow: float
        RmsWindow value in SPES data file
    rmswindowstart: float
        RmsWindowStart value in SPES data file
    filter_spearman_r: float
        Threshold Spearman'r for SPES responses
    filter_spearman_p: float
        Threshold Spearman'p for SPES responses
    filter_zscore: float
        Threshold zscore for SPES responses. This is helpful to remove outliers, for eaxmple contacts nearby the
        stimulation site exhibiting stimulation artefacts.
    percentile_threshold: float
        Percentile threshold. Will keep responses higher than this threshold. Default Q3 (75%)

    Returns
    -------

    """

    RAW_DATA, RAW_DATA_NATIVE, SUBJECTS_DIR, SUBJECTS_DIR_NATIVE = paths.set_paths(paths.HOSTNAME)
    cceps, sheetname = inout.load_spes(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'SPES', 'SPES.xls'),
                                       sheetname='SEEG', protocol=protocol, lowfreq=lowfreq, highfreq=highfreq,
                                       rmswindow=rmswindow, rmswindowstart=rmswindowstart)
    if filter_zscore is not None:
        # filter out outliers > XX SD (mostly channels adjacent to stimulation site, but who knows what other artefacts...
        cceps = cceps[zscore(cceps['mean_rms']) < filter_zscore].reset_index(drop=True)

    if percentile_threshold is not None:
        q3 = np.percentile(cceps['mean_rms'].values, percentile_threshold)
        cceps = cceps[cceps['mean_rms'] > q3].reset_index(drop=True)
    else:
        q3 = 0

    if (filter_spearman_r is not None) and (filter_spearman_p is not None):
        cceps = cceps[(cceps['pvalue'] < filter_spearman_p) & (cceps['correlation'] > filter_spearman_r)].reset_index(
            drop=True)

    # filter out responses on stimulation contacts
    cceps = cceps[cceps.apply(lambda x: x['RespContact'] not in x['StimContact'], axis=1)].reset_index(drop=True)
    return cceps, q3


def average_by_structure(subj, flow='outbound', localization='most_likely', protocol='SPES', lowfreq=0, highfreq=0,
                         rmswindow=100, rmswindowstart=10,
                         filter_spearman_r=0.5, filter_spearman_p=0.05, filter_zscore=3, percentile_threshold=75):
    RAW_DATA, RAW_DATA_NATIVE, SUBJECTS_DIR, SUBJECTS_DIR_NATIVE = paths.set_paths(paths.HOSTNAME)
    cceps, q3 = get_cceps(subj, protocol=protocol, lowfreq=lowfreq, highfreq=highfreq, rmswindow=rmswindow,
                          rmswindowstart=rmswindowstart,
                          filter_spearman_r=filter_spearman_r, filter_spearman_p=filter_spearman_p,
                          filter_zscore=filter_zscore, percentile_threshold=percentile_threshold)
    contacts = pd.read_excel(os.path.join(SUBJECTS_DIR_NATIVE, subj, 'Contact_coordinates.xlsx'))
    # filter out contacts in white matter, or unknown locations
    contacts = contacts[~contacts[localization].isin(
        ['Unknown', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter'])].reset_index(drop=True)

    structures = list(contacts[localization].unique())
    groups = contacts.groupby(localization)
    stim_pairs, contact1, contact2 = get_stim_contact(cceps)
    for s in structures:
        # contacts within structure s
        cnames = list(groups.get_group(s).name.values)
        # replace contacts by structures in cceps table
        cceps.loc[cceps['StimContact'].str.contains('|'.join(cnames)), 'StimContact'] = s
        cceps.loc[cceps['RespContact'].str.contains('|'.join(cnames)), 'RespContact'] = s
    struct_pairs = list(itertools.permutations(structures, 2))
    table_header = ['structure1', 'structure2', 'mean_rms', 'std_rms', 'q3normed_mean_rms', 'q3normed_std_rms',
                    'median_rms', 'mad_rms', 'q3normed_median_rms', 'q3normed_mad_rms', 'npairs']
    connectivity_table = pd.DataFrame(columns=table_header)
    for sp in struct_pairs:
        selection = cceps[(cceps.StimContact == sp[0]) & (cceps.RespContact == sp[1])]
        if selection.shape[0] > 0:
            connectivity_table = connectivity_table.append(
                pd.DataFrame([[
                    sp[0],
                    sp[1],
                    selection.mean_rms.mean(),
                    selection.mean_rms.std(),
                    (selection.mean_rms / q3).mean(),
                    (selection.mean_rms / q3).std(),
                    selection.mean_rms.median(),
                    selection.mean_rms.mad(),
                    (selection.mean_rms / q3).median(),
                    (selection.mean_rms / q3).mad(),
                    selection.shape[0]]],
                    columns=table_header,
                ),
                ignore_index=True)
    return connectivity_table


def table_to_adjacency_matrix(table, label_columns, value_column):
    labels = list(set(table[label_columns].values.ravel()))
    labels.sort()
    adj_mat = np.zeros((len(labels), len(labels))) * np.NAN
    for index, row in table.iterrows():
        i = labels.index(row[label_columns[0]])
        j = labels.index(row[label_columns[1]])
        adj_mat[i][j] = row[value_column]
    return adj_mat, labels
