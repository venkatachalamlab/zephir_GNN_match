####################################################################################
# This module is to build the graph by treating the neuron as node, embed with nodes features and edge features
# The label comes from the annotation's label
####################################################################################

import numpy as np
import h5py
import pandas as pd
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import time
import matplotlib.pyplot as plt
from scipy.stats import mode
import torch
from torch_geometric.data import Data
####################################################################################
# create nodes features
####################################################################################
def get_neuron_3D_feartures(seg,img_orig):
    props = regionprops_table(seg.astype(int), intensity_image = img_orig,
                                              properties=('centroid',
                                                           'area_filled',
                                                           'intensity_mean',
                                                          'centroid',
                                                          # 'centroid_weighted',
                                                          'coords',
                                                          'slice',
                                                          'label')
                             )
    df_3D = pd.DataFrame(props)
    df_3D['slice_depth'] = df_3D['slice'].apply(lambda x: x[0].stop - x[0].start)
    df_3D = df_3D.drop(columns=['slice'])
    df_3D = df_3D.rename(columns={"area_filled":"volume",
                                  'intensity_mean':'intensity_mean_3D'})
    return df_3D


def get_neuron_2D_feartures(seg,img_orig):
    '''
    for the maximum projection of each neuron, obtain the features
    '''
    neuron_ID_list = list(np.unique(seg).astype(int))
    neuron_ID_list.remove(0)
    props_df_list = []
    for neuron_ID in neuron_ID_list:

        neuron_ind = seg == neuron_ID
        seg_ID_2d = neuron_ID*np.max(neuron_ind,axis = 0).astype(int)
        img_ID_2d = np.max(neuron_ind*img_orig,axis = 0)
        props = regionprops_table(seg_ID_2d, 
                          intensity_image = img_ID_2d,
                                  properties=(
                                             'orientation',
                                             'axis_major_length',
                                             'axis_minor_length',
                                             'area_filled', 
                                             'eccentricity', 
                                             'perimeter', 
                                             'intensity_mean',
                                             'label',
                                             'eccentricity',
                                             'centroid',
                                             # 'centroid_weighted',
                                            
                        )
                     )

        props_df = pd.DataFrame(props)
        props_df_list.append(props_df)
        
    df_2D = pd.concat(props_df_list, ignore_index=True)
    df_2D['axis_ratio'] = df_2D['axis_minor_length']/df_2D['axis_major_length']
    df_2D = df_2D.rename(columns={"area_filled":"area_zmax",
                                  'intensity_mean':'intensity_mean_2D',
                                  'centroid-0': 'centroid_2d-0',
                                  'centroid-1': 'centroid_2d-1'
                                 })
    return df_2D



####################################################################################
# create nodes features
####################################################################################


def get_indices_arr(neuron_ID_arr,nodes_arr):
    '''
    neuron_ID_arr: the referecen array that are used to find the indices of each values
    nodes_arr: the target array that needs to find each number's indices 
    '''
    value_to_index = {value: idx for idx, value in enumerate(neuron_ID_arr)}
    indices = [value_to_index[node] for node in nodes_arr]
    return np.array(indices)

def vector_orientation(orientation_arr):
    '''
    convert the orientation(-pi,pi) into unit vecot
    '''
    arr = np.array([np.cos(orientation_arr), - np.sin(orientation_arr)])
    return np.transpose(arr,(1,0))

def search_nearby_nodes(df_nodes_features, num_nearest):
    '''
    find num nearesat neurons by distance 
    return the indices based on the df_nodes_features['label']
    update as neighbour with which neuron_ID is nearby
    '''
    neuron_centroid = np.array(df_nodes_features[['centroid-0','centroid_2d-0','centroid_2d-1']])
    neuron_ID_arr = np.array(df_nodes_features['label'])
    target_point = neuron_centroid[0]
    tree = KDTree(neuron_centroid)
    _, indices = tree.query(neuron_centroid, k=num_nearest+1)
    index = indices[:,1:] # exclude itself
    df_nodes_features['neighbour'] = list(neuron_ID_arr[index])
    return df_nodes_features, index

def get_pair_nodes(df_nodes_features,num_nearest,index):
    '''
    return the edge with pair of nodes
    '''
    neuron_ID_arr = np.array(df_nodes_features['label'])
    repeated_array = np.repeat(neuron_ID_arr[:, np.newaxis], num_nearest, axis=1)
    arr = np.array([repeated_array,neuron_ID_arr[index]])
    pair_nodes = np.transpose(arr,(1,2,0)).reshape(-1,2)
    return pair_nodes

def get_edges_features(df_nodes_features,pair_nodes):
    '''
    calculate the euclidean distance, euclidean vector, and angle between two neurons
    '''
    neuron_ID_arr = np.array(df_nodes_features['label'])
    neuron_centroid = np.array(df_nodes_features[['centroid-0','centroid_2d-0','centroid_2d-1']])
    
    nodes_start_indices = get_indices_arr(neuron_ID_arr,pair_nodes[:,0])
    nodes_end_indices = get_indices_arr(neuron_ID_arr,pair_nodes[:,1])
    coords_start = neuron_centroid[nodes_start_indices]
    coords_end = neuron_centroid[nodes_end_indices]
    euclidean_vector = coords_end - coords_start
    euclidean_distances = np.sqrt(np.sum(euclidean_vector*euclidean_vector,axis=1))

    nodes_start_ore = np.array(df_nodes_features['orientation'])[nodes_start_indices]
    nodes_end_ore = np.array(df_nodes_features['orientation'])[nodes_end_indices]
    pair_orientation = (np.sum(vector_orientation(nodes_start_ore) * vector_orientation(nodes_end_ore), axis = 1))

    df_edges_features = pd.DataFrame(np.transpose(np.array([euclidean_vector[:,0],euclidean_vector[:,1],euclidean_vector[:,2],
                                        euclidean_distances,pair_orientation]),(1,0)),
                                    columns=['euclidean_vector-0','euclidean_vector-1','euclidean_vector-2','euclidean_abs_dist','pair_orientation'])
    df_edges_features.insert(0, 'pair_nodes', [tuple(pair) for pair in pair_nodes]) 
    return df_edges_features, nodes_start_indices,nodes_end_indices





####################################################################################
# create nodes features
####################################################################################









################################################################################################################
    # The following module is to match the segmentation neurons with annotation ID
################################################################################################################
def load_annotations_h5_t_idx(file_name,t_idx):
    '''
    Load annotation at all time 
    '''
    with h5py.File(file_name, 'r') as hdf: 
        dfs = []
        for key in hdf.keys():
            dataset = hdf[key]
            df = pd.DataFrame(dataset[:],columns=[key])
            dfs.append(df)

        combined_df = pd.concat(dfs, axis = 1)
    hdf.close()
    combined_df_t_idx = combined_df[combined_df['t_idx'] == t_idx]
    return combined_df_t_idx
    
def get_abs_pos(combined_df_t_idx,labels_shape):
    '''
    Obtain the absolutely coordinates annotations at the time t_idx with the segmented size of labels_shape from the normalized coordinates
    !!! Important notice: the absolute coordinates starting from 0, then it should be multiplied by (labels_shape-1)
    '''
    pos = np.array(combined_df_t_idx[['z','y','x']])
    abs_pos = (np.round(pos * (np.array(labels_shape)-1))).astype(int)
    return abs_pos



def get_unique_ID_Neuron(abs_pos,label_z):
    ## find the ID's segmentatino values in the nearby [-1,1]
    z,y,x = abs_pos[:,0], abs_pos[:,1], abs_pos[:,2]
    most_frequent = label_z[z,y,x]
    return most_frequent


def if_unique_ID(most_frequent):
    '''
    To find if this segmentation only returns one unique annotation ID
    '''
    unique_values, inverse_indices, counts = np.unique(most_frequent, return_inverse=True, return_counts=True)
    is_unique = counts[inverse_indices] == 1
    return is_unique


def annotation_seg_ID(annotation_path,t_idx,seg,train_val):
    '''
    To assign the world line ID to the uniquely segmented neuron
    '''
    combined_df_t_idx = load_annotations_h5_t_idx(annotation_path,t_idx)
    abs_pos = get_abs_pos(combined_df_t_idx,seg.shape)
    most_frequent = get_unique_ID_Neuron(abs_pos,seg)
    is_unique = if_unique_ID(most_frequent)
    most_frequent[is_unique==False] = 0
    combined_df_t_idx['seg_ID'] = most_frequent
    combined_df_t_idx[['global_z','global_gy','global_gx' ]] = abs_pos
    
    
    annotation_df = combined_df_t_idx[['worldline_id', 'seg_ID']]
    annotation_df = annotation_df.rename(columns={'seg_ID':'label'})
    annotation_df['worldline_id'] = annotation_df['worldline_id'] +1
    
    
    
    return annotation_df[annotation_df['label']>0]




################################################################################################################
    # The following module is to produce nodes & edges features in the graph
################################################################################################################
def produce_nodes_features(seg,img_orig,annotation_df,train_val):
    img_norm = normalize(img_orig,pmin=3,pmax=99.8)
    img_norm = img_norm/np.max(img_norm)
    df_3D = get_neuron_3D_feartures(seg,img_norm)
    df_2D = get_neuron_2D_feartures(seg,img_orig)
    df_nodes_features = pd.merge(df_2D, df_3D, on='label', how='inner')
    
    if train_val == 'train':
        label_to_worldline_id = dict(zip(annotation_df['label'], annotation_df['worldline_id']))
        df_nodes_features['worldline_id'] = df_nodes_features['label'].map(label_to_worldline_id).fillna(0)
        
    return df_nodes_features
    


def produce_edges_features(df_nodes_features, num_nearest):
    df_nodes_features, index = search_nearby_nodes(df_nodes_features, num_nearest)
    pair_nodes = get_pair_nodes(df_nodes_features,num_nearest,index) 
    df_edges_features,nodes_start_indices,nodes_end_indices = get_edges_features(df_nodes_features,pair_nodes)
    return df_edges_features,nodes_start_indices,nodes_end_indices









































