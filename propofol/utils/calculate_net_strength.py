import pandas as pd
import numpy as np
import sys
import os
from scipy import stats
import pingouin as pg
import scipy.io as sio
import random
import math
from fc_utils.load_fc import load_fc
from fc_utils.get_edge_vals import get_edge_vals
from propofol.utils.subjects import subjects
import os

cwd = os.getcwd()


BASE_DATA_DIR = '/Users/taylorchamberlain/code/propofol_paper/propofol/data'


def make_sym_matrix(n,vals):
    m = np.zeros([n,n], dtype=np.double)
    xs,ys = np.triu_indices(n,k=1)
    m[xs,ys] = vals
    m[ys,xs] = vals
    m[ np.diag_indices(n) ] = 0
    return m


def get_attn_net_strength(X, nodes_to_drop, use_average=True, use_engage=False, use_greene=False,  greene_index=None):
    # here  we take just half of symmetric matrix
    if use_greene:
        greeneCPM_load = sio.loadmat('/Users/taylorchamberlain/code/fc_ts_utils/fc_utils/network_definitions/gFnets_p01_hcp.mat')
        gf_posedges = greeneCPM_load['task_pos_edge'][greene_index,:,:]
        gf_negedges = greeneCPM_load['task_neg_edge'][greene_index,:,:]
        # nodes_to_drop = None
        # load greene net
        nodes_missing_in_green =  [108, 114, 117, 128, 238, 242, 248, 249, 265]
        # convert greene into 268 x 268
        missing_so_far_i, missing_so_far_j = 0, 0
        greene_268_pos = np.full([268, 268], np.nan)
        greene_268_neg = np.full([268, 268], np.nan)

        for i in range(268):
            if i in nodes_missing_in_green:
                missing_so_far_i += 1
            missing_so_far_j = 0
            for j in range(268):
                if j in nodes_missing_in_green:
                    missing_so_far_j+=1
                if i not in nodes_missing_in_green and j not in nodes_missing_in_green:
                    greene_268_pos[i, j] = gf_posedges[i-missing_so_far_i, j-missing_so_far_j]
                    greene_268_neg[i, j] = gf_negedges[i-missing_so_far_i, j-missing_so_far_j]

        posedges = np.nan_to_num(greene_268_pos)
        negedges = np.nan_to_num(greene_268_neg)

    elif use_engage:
        load = sio.loadmat('../network_definitions/auditory_engagement_shen_mask.mat')
        posedges = load['pos']
        negedges = load['neg']
        nodes_to_drop = None
    else:
        posedges = np.loadtxt('/Users/taylorchamberlain/code/fc_ts_utils/fc_utils/network_definitions/sa_cpm/sa_cpm_pos_mask.txt')
        negedges = np.loadtxt('/Users/taylorchamberlain/code/fc_ts_utils/fc_utils/network_definitions/sa_cpm/sa_cpm_neg_mask.txt')

    if nodes_to_drop:
        num_nodes = posedges.shape[1]
        indices_minus_nodes_to_drop = [i for i in range(num_nodes) if i not in nodes_to_drop]
        posedges = posedges[ :, indices_minus_nodes_to_drop]
        posedges = posedges[ indices_minus_nodes_to_drop, :]

        num_nodes = negedges.shape[1]
        negedges = negedges[ :, indices_minus_nodes_to_drop]
        negedges = negedges[ indices_minus_nodes_to_drop, :]

    # only take half of the symmetric matrix
    # even tho it makes no diff on end result
    iu1 = np.triu_indices(X.shape[0])
    X[iu1] = np.nan

    pe=X[posedges.astype(bool)]
    ne=X[negedges.astype(bool)]

    if use_average:
        pos_strength = np.nanmean(pe)
        neg_strength = np.nanmean(ne)
    else:
        pos_strength = np.nansum(pe)
        neg_strength = np.nansum(ne)

    return pos_strength, neg_strength

def get_edge_vals(mat, exclude_diag=True):
    size = mat.shape[0]
    index= np.triu_indices(size)
    if exclude_diag:
        index= np.triu_indices(size, 1)
    else:
        index= np.triu_indices(size)

    return mat[index]


def get_random_attn_net_strength(posedges, negedges, X, nodes_to_drop, use_average=True):
    if nodes_to_drop:
        num_nodes = posedges.shape[1]
        indices_minus_nodes_to_drop = [i for i in range(num_nodes) if i not in nodes_to_drop]

        posedges = posedges[ :, indices_minus_nodes_to_drop]
        posedges = posedges[ indices_minus_nodes_to_drop, :]

        num_nodes = negedges.shape[1]
        negedges = negedges[ :, indices_minus_nodes_to_drop]
        negedges = negedges[ indices_minus_nodes_to_drop, :]

    pe=X[posedges.astype(bool)]
    ne=X[negedges.astype(bool)]



    if use_average:
        pos_strength = np.nanmean(pe)
        neg_strength = np.nanmean(ne)
    else:
        pos_strength = np.nansum(pe)
        neg_strength = np.nansum(ne)

    return pos_strength, neg_strength



def load_fc_data(sub, session, task, run, data_dir, nodes_to_drop=None, subset_fc=None):
    path = f'{BASE_DATA_DIR}/{data_dir}/{sub}_{session}_{task}_{run}_LPI_000.netcc'
    if os.path.exists(path):
        fc = load_fc(path)

        if nodes_to_drop:
            num_nodes = fc.shape[1]
            indices_minus_nodes_to_drop = [i for i in range(num_nodes) if i not in nodes_to_drop]

            fc = fc[:, indices_minus_nodes_to_drop]
            fc = fc[indices_minus_nodes_to_drop, :]

        return fc
    else:
        # TODO:  change this to log missing runs in a file
        pass

def load_network_df(data_dir,
                    nodes_to_drop=None,
                    use_greene=False,
                    use_average=True,
                    greene_index=None,
                    use_engage=False):
    runs = ['01', '02', '03', '04']
    tasks = ['rest', 'movie']
    # tasks = ['rest']
    sessions = ['01']

    pos_list = []
    neg_list = []
    run_list_neg = { '01':[], '02':[], '03':[], '04':[] }
    run_list_pos = { '01':[], '02':[], '03':[], '04':[] }

    data_dict = {
        'subject':[],
        'task':[],
        'sedation_level':[],
        'pos_network':[],
        'neg_network':[],
    }


    for sub in subjects:
        for session in sessions:
            for task in tasks:
                for run in runs:
                    fc = load_fc_data(sub, session, task, run, data_dir, nodes_to_drop=nodes_to_drop)
                    if fc is not None:
                        pos, neg = get_attn_net_strength(fc,
                                                         nodes_to_drop=nodes_to_drop,
                                                         use_average=use_average,
                                                         use_engage=use_engage,
                                                         use_greene=use_greene,
                                                         greene_index=greene_index)
                        pos_list.append(pos)
                        neg_list.append(neg)
                        run_list_neg[run].append(neg)
                        run_list_pos[run].append(pos)
                        data_dict['subject'].append(sub)
                        data_dict['sedation_level'].append(run)
                        data_dict['task'].append(task)
                        data_dict['neg_network'].append(neg)
                        data_dict['pos_network'].append(pos)

    df = pd.DataFrame(data_dict)
    motion_df = pd.read_csv(f'{BASE_DATA_DIR}/ss_out_review_compiled/ss_out_review_compiled_censor_frames.csv')
    motion_df['subject'] = motion_df['sub']
    motion_df['sedation_level'] = motion_df['run'].apply(lambda x: "0" + str(x))
    df = pd.merge(df, motion_df, on=['subject', 'sedation_level', 'task'])
    return df


def run_t_tests(df, task='rest', network='pos_network'):
    task_df = df[df['task']==task]
    one = list(task_df[task_df['sedation_level']=='01']['subject'])
    three = list(task_df[task_df['sedation_level']=='03']['subject'])

    has_one_and_three_sub_ids = list(set(one) & set(three))
    task_df = task_df[task_df['subject'].isin(has_one_and_three_sub_ids)]

    awake_list_pos = list(task_df[task_df['sedation_level']=='01'][network])
    deep_list_pos = list(task_df[task_df['sedation_level']=='03'][network])

    res = stats.ttest_rel(awake_list_pos, deep_list_pos)
    print(f'{network} result for task: {task}')
    n = len(awake_list_pos)

    return res, n
