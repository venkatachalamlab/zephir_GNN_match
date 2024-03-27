
import h5py
import torch
import numpy as np
from .Build_graph_seg import *
from .GNN_model import *
# from Build_graph_seg import *
# from GNN_model import *
# from Graph_data import *
import pathlib
import time
import concurrent.futures


def get_volume_at_frame(file_name,t_idx):
    '''
    Get the 3D original volume at frame t_idx from the h5 file_name
    '''
    with h5py.File(file_name, 'r') as f:
        img_original = np.array(f['data'][t_idx:t_idx+1])
        mask = None
    f.close()
    return img_original,mask 

def produce_graph_data(df_nodes_features, df_edges_features,nodes_start_indices,nodes_end_indices,train_val):
    # selected_nodes_features = ['orientation', 'axis_major_length', 'axis_minor_length',
    #        'area_zmax', 'eccentricity', 'perimeter', 'intensity_mean_2D',
    #        'centroid_2d-0', 'centroid_2d-1', 'axis_ratio', 'centroid-0',
    #        'centroid-1', 'centroid-2', 'volume', 'intensity_mean_3D', 'slice_depth']
    selected_nodes_features = [
            'centroid_2d-0', 'centroid_2d-1', 'centroid-0',
            'axis_major_length', 'axis_minor_length',
             'axis_ratio', 
        'eccentricity', 
           'centroid-1', 'centroid-2', 
            'slice_depth',
            ]
    selected_edges_features = ['euclidean_vector-0', 'euclidean_vector-1','euclidean_vector-2','euclidean_abs_dist', 'pair_orientation']
    node_features = torch.tensor(np.array(df_nodes_features[selected_nodes_features].values), dtype=torch.float)
    edge_index = torch.tensor(np.array([nodes_start_indices, nodes_end_indices]), dtype=torch.long)
    edge_features = torch.tensor(np.array(df_edges_features[selected_edges_features].values), dtype=torch.float)
 
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    # data.y = torch.tensor(np.array(df_nodes_features['worldline_id']))
    if train_val == 'train':
        data.y = torch.tensor(np.array(df_nodes_features['worldline_id']))
    elif train_val == 'eval':
        ## in the prediction mode: no data.y will be produced and the worldline
        pass
    return data


def produce_data_graph(seg,img_orig,annotation_df,num_nearest,train_val):
    df_nodes_features = produce_nodes_features(seg,img_orig,annotation_df,train_val)
    df_edges_features,nodes_start_indices,nodes_end_indices = produce_edges_features(df_nodes_features, num_nearest)
    data = produce_graph_data(df_nodes_features, df_edges_features,nodes_start_indices,nodes_end_indices,train_val)
    return data


def get_data_generator(t_idx, ch, num_nearest,img_h5_path,seg_h5_path,annotation_path,train_val):
    '''
    num_nearest: search for the neareast 5 points
    '''

    img_original,_= get_volume_at_frame(img_h5_path,t_idx)
    img_orig = img_original[0,ch]


    f = h5py.File(seg_h5_path+str(t_idx)+'.h5', 'r')
    seg = f['label'][:]
    f.close()

    if train_val == 'train':
        annotation_df = annotation_seg_ID(annotation_path,t_idx,seg,train_val)
    elif train_val == 'eval':
        annotation_df = []

    data = produce_data_graph(seg,img_orig,annotation_df,num_nearest,train_val)
    return data



def save_pandas_h5(h5_filename,pandas_df):
    with h5py.File(h5_filename, 'w') as h:
        for k, v in pandas_df.items():
            h.create_dataset(k, data=np.array(v.values))
    h.close()