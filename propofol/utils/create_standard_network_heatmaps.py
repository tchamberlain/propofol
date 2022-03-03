import numpy as np
import pandas as pd
# from subjects import subjects
from colour import Color
from matplotlib.colors import LinearSegmentedColormap
import matlab.engine
import scipy.io as sio
import seaborn as sns
import os
from propofol.utils.calculate_net_strength import run_t_tests
import matplotlib.pyplot as plt

def make_ramp( ramp_colors ):
    color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )
    plt.figure( figsize = (15,3))
    plt.imshow( [list(np.arange(0, len( ramp_colors ) , 0.1)) ] , interpolation='nearest', origin='lower', cmap= color_ramp )
    plt.xticks([])
    plt.yticks([])
    return color_ramp

def load_fc_data(sub, session, task, run, data_dir, nodes_to_drop=None, subset_fc=None):
        base_path = f'./data/{data_dir}'
        path = f'{base_path}/{sub}_{session}_{task}_{run}_LPI_000'
        if os.path.exists(f'{path}.netcc'):
            paths = [path]
            fc, ts = load_mats([sub], paths, condition='', subset_fc=subset_fc)

            if nodes_to_drop:
                num_nodes = fc.shape[1]
                indices_minus_nodes_to_drop = [i for i in range(num_nodes) if i not in nodes_to_drop]

                fc = fc[:, :, indices_minus_nodes_to_drop]
                fc = fc[:, indices_minus_nodes_to_drop, :]

            return fc
        else:
            print('file not found')


def get_network_mat(sub_mat):
    # for sub in sub_list_fc:
    eng = matlab.engine.start_matlab()
    eng.addpath("./")
    sio.savemat('./sub_fc_mat.mat', mdict={'fc': sub_mat})

    network_mat = eng.conv2network()
    network_mat = np.asarray(network_mat)
    return network_mat

def get_edge_vals(mat, exclude_diag=True):
    size = mat.shape[0]
    index= np.triu_indices(size)
    if exclude_diag:
        index= np.triu_indices(size, 1)
    else:
        index= np.triu_indices(size)

    return mat[index]


def make_sym_matrix(n,vals):
    m = np.zeros([n,n], dtype=np.double)
    xs,ys = np.triu_indices(n,k=1)
    m[xs,ys] = vals
    m[ys,xs] = vals
    return m



def get_diag_from_flat(flat):
    return flat[[0, 8, 15, 21, 26, 30, 33, 35]]

def rm_diag_from_flat(flat):
    return np.delete(flat, [0, 8, 15, 21, 26, 30, 33, 35], None)


def from_flat_to_mat(flat):
    # rm diag
    no_diag = rm_diag_from_flat(flat)
    # reshape into mat
    sym = make_sym_matrix(8, no_diag)
    # put back diag
    np.fill_diagonal(sym, get_diag_from_flat(flat))
    iu1 = np.triu_indices(8, 1)
    sym[iu1] = np.nan
    return sym

def create_standard_network_heatmaps(task,
                                     df,
                                     color_range=(-2, 2),
                                     ax=None,
                                     color_bar=False,
                                     cbar_ax=None,
                                     y_tick=True):
    p_vals = []
    stats = []
    for flat_mat_index in range(36):
        res, n = run_t_tests(df[df['flat_mat_index']==flat_mat_index], task=task, network='sum_of_strengths')
        p_vals.append(res.pvalue)
        stats.append(res.statistic)

    p_vals = np.array(p_vals)
    p_val_map = from_flat_to_mat(p_vals)


    stats = np.array(stats)
    stats_map = from_flat_to_mat(stats)



    net_names= ['medial frontal',
        'frontal parietal',
        'DMN',
        'subcortical/cerebellar',
        'motor',
        'visual I',
        'visual II',
        'visual association',]

    labels = []
    p = .05
    for row in (p_val_map < p):
        label_row = ['*' if i else '' for i in row]
        labels.append(label_row)
    labels = np.array(labels)

    if y_tick:
        yticklabels= net_names
    else:
        yticklabels= ['' for i in range(len(net_names))]

    sns.set_style("white")
    cmap = sns.diverging_palette(189, 37, s=100, l=66, as_cmap=True)
    sns.set_context("notebook", font_scale=3.5, rc={"lines.linewidth": 2.5})
    sns.set_theme(style="white", font_scale=3.5,rc={"text.color": 'black',
                                                    'axes.labelcolor': 'black',
                                                  'font.family': 'serif',
                                                  'font.serif':'Helvetica'})



    sns.heatmap(stats_map,
                ax = ax,
                xticklabels=net_names,
                annot_kws={"size": 18},
                yticklabels =yticklabels,
                cmap=cmap,
                vmin=color_range[0], vmax=color_range[1],
                annot=labels,
                fmt="s",
                cbar=color_bar,
                cbar_ax = cbar_ax,
               linewidths=4)
    return stats_map, p_val_map
