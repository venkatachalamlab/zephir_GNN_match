from .GNN_model import *
from scipy.spatial import distance

def save_pandas_h5(save_h5_path, df):
    with h5py.File(save_h5_path, 'w') as hdf:
        for column in df.columns:
            data = df[column].to_numpy()
            if data.dtype == object:
                data = data.astype(h5py.string_dtype())
            hdf.create_dataset(column, data=data)
    hdf.close()

def load_model_args(model_path, nearby_search_num_list,with_AM,device,train_val):
    model_all = {}
    for nearby_search_num in nearby_search_num_list:
        model_all[nearby_search_num] = NodeLevelGNN(nearby_search_num, with_AM, device,train_val).to(device)
        model_all[nearby_search_num].load_state_dict(torch.load(model_path)['model'])
    return model_all


def find_lines_with_word(file_path, word):
    lines_with_word = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # print(line_number,line)
            if word in line:
                
                lines_with_word.append(line)
    return lines_with_word


def get_current_refe_frames(lines_with_test):
    pair_t1_t2 = []
    for i in range(len(lines_with_test)):
        pattern = r'Frame #(\d+).*Parent #(\d+)'
        matches = re.search(pattern, lines_with_test[i])
        if matches is not None:
            # print([int(matches.group(1)),int(matches.group(2))],lines_with_test[i])
            pair_t1_t2.append([int(matches.group(1)),int(matches.group(2))])
    return np.array(pair_t1_t2)


def get_pair_data4(t1,t2,device,path_graph):

    data1 = torch.load(path_graph+str(t1)+'.pt')
    data2 = torch.load(path_graph+str(t2)+'.pt')

   
    return data1.to(device),data2.to(device)


def get_AM_mask(data1,data2,nearby_search_num,with_AM,device,train_val):
    # nearby_search_num = 10
    coord = data1.x[:,0:3]
    coords2 = data2.x[:,0:3]
    # distance_matrix = cdist(coord,coords2)
    # nearest_indices = np.argsort(distance_matrix, axis=1)[:, :nearby_search_num]
    # mask = np.zeros_like(distance_matrix, dtype=bool)
    distance_matrix = torch.cdist(coord,coords2)
    nearest_indices = torch.argsort(distance_matrix, dim=1)[:, :nearby_search_num]
    mask = torch.zeros_like(distance_matrix, dtype=bool).to(device)
    for row_idx, col_indices in enumerate(nearest_indices):
        mask[row_idx, col_indices] = 1

    if with_AM:
        AM = get_ground_truth(data1,data2)
        AM_mask = (mask + 1*(AM>0))>0
    else:
        # AM_mask = (torch.tensor(mask))>0
        AM_mask = mask > 0
    if train_val== 'train':
        AM_mask[data1.y==0] = 0
    # elif train_val == 'eval' and 'y' in data.keys():
    #     if np.max(data.y) == 1:
    #         AM_mask[data1.y==0] = 0
    return AM_mask


def get_edge_label(data1,data2,nearby_search_num,with_AM,device,train_val):
    AM = get_ground_truth(data1,data2)
    AM_mask = get_AM_mask(data1,data2,nearby_search_num,with_AM,device,train_val)
    if with_AM:
        AM_label = AM_mask + AM
        edge_label = AM_label[AM_mask>0]-1
    else:
        edge_label =  (AM*AM_mask)[AM_mask>0]
    return edge_label


def get_ground_truth(data1,data2):
    eff_label = (data1.y.unsqueeze(1) + data2.y.unsqueeze(0)) > 0  ## to exclude label=0
    AM = ((data1.y.unsqueeze(1) - data2.y.unsqueeze(0)) == 0)*(eff_label)*1
    return AM


def get_AM_pred(model,data1,data2,threshold,nearby_search_num,with_AM,device,train_val):
    with torch.no_grad():  
        all_match_scores1 = model(data1, data2)

    all_match_scores1_softmax = torch.softmax(all_match_scores1,dim = 1)
    all_match_scores_m_values, all_match_scores_m  = torch.max(all_match_scores1_softmax,dim = 1)
    all_match_scores_m[all_match_scores_m_values<threshold] = 0

    AM_mask = get_AM_mask(data1,data2,nearby_search_num,with_AM,device,train_val)
    edge_label_matrix = torch.zeros(AM_mask.shape,dtype = torch.long).to(device)

    edge_label_matrix[AM_mask>0] =  all_match_scores_m
    ind1 = torch.sum(edge_label_matrix,dim = 1) > 1
    ind2 = torch.sum(edge_label_matrix,dim = 0) > 1
    edge_label_matrix[ind1,:] = 0
    edge_label_matrix[:,ind2] = 0
    return edge_label_matrix


def switch_prediction(model_10,data1,data2,threshold,nearby_search_num,with_AM,device,train_val):
    edge_label_matrix1 = get_AM_pred(model_10,data1,data2,threshold,nearby_search_num,with_AM,device,train_val)
    edge_label_matrix2 = get_AM_pred(model_10,data2,data1,threshold,nearby_search_num,with_AM,device,train_val)
    edge_label_matrix = edge_label_matrix1 * (edge_label_matrix2.T)
    return edge_label_matrix 

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



def model_match_patial_neurons(path_graph,t_frame,model_all,img_shape,threshold,nearby_search_num_list,with_AM,device,train_val):
    #### The following module is to produce the pandas dataframe

    measure_matrix = np.array([0,0,0,0])

    df_all =  pd.DataFrame()

    for i in tqdm(range(len(t_frame)-1)):
    # for i in tqdm(range(10)):
        t1 = t_frame[i]
        t2 = t_frame[i+1]

        # data1, data2 = get_pair_data4(int(t1),int(t2),device)
        data1, data2 = get_pair_data4(t1,t2,device,path_graph)

        ### To obatain the predicted class in the data2 and built the partial annotations at frame t2
        edge_label_matrix = torch.ones(len(data1.x),len(data2.x)).to(device)
        for nearby_search_num in nearby_search_num_list: 
            edge_label_matrix_model = switch_prediction(model_all[nearby_search_num],data1,data2,threshold,nearby_search_num,with_AM,device,train_val)
            edge_label_matrix = edge_label_matrix * edge_label_matrix_model

        #### Getting the class in data1 from the annotations at frame t1 and the segmentation
        edge_label_matrix[data1.y==0,:] = 0 ## For the reference frame, if the neuron is not uniquely segmented, then ignore
        AM_mask = get_AM_mask(data1,data2,5,with_AM,device,train_val)

        
        
        ### Get the predicted class in the data2 from the predicted matrix
        ind = torch.where((edge_label_matrix*(AM_mask > 0))>0)
        # data2_y = torch.zeros(len(data2.x)).to(torch.int).to(device)
        # data2_y[ind[1]] = (data1.y[ind[0]]).to(torch.int)
        selected_index = torch.tensor([8,7,2],dtype = torch.long).to(device) ## ## the coordinates index
        # coords_2 = data2.x[selected_index]

        ind = torch.where((edge_label_matrix*AM_mask)>0)
        norm_factor = (torch.tensor(img_shape) - 1).to(device)
        parent_coords = (data1.x[ind[0]][:,selected_index]/norm_factor).cpu()
        child_coords = (data2.x[ind[1]][:,selected_index]/norm_factor).cpu()
        
        if train_val == 'train':
            df_t2 = {
                't_idx': t2,
                'x': child_coords[:,0],
                'y': child_coords[:,1],
                'z': child_coords[:,2],
                'parent_id': t1,
                'parent_coords_x': parent_coords[:,0],
                'parent_coords_y': parent_coords[:,1],
                'parent_coords_z': parent_coords[:,2],
                'worlidline_id':(data1.y[ind[0]]-1).cpu().to(torch.int)
                 }
            
        elif train_val == 'eval':
            df_t2 = {
                't_idx': t2,
                'x': child_coords[:,0],
                'y': child_coords[:,1],
                'z': child_coords[:,2],
                'parent_id': t1,
                'parent_coords_x': parent_coords[:,0],
                'parent_coords_y': parent_coords[:,1],
                'parent_coords_z': parent_coords[:,2],
                 }
            
        df = pd.DataFrame.from_dict(df_t2)
        df_all = pd.concat([df_all,df])
    return df_all








def find_merged_index_graphdata(seg_path,t1,img_shape,annotation):
    '''
    seg_path: the path used to save the segmentation for each time index
    t1: the parent time index
    img_shape: the 3D volume size 
    annotation: the newest annotation from the zephir
    return: the merged neuron index of the graph data
    '''
    h = h5py.File(seg_path + str(t1) +'.h5', 'r')
    seg = h['label'][:]
    h.close()
    rescale = np.array(img_shape)  - 1
    annotation_t_idx = annotation[annotation['t_idx'] == t1]
    coords_orig = annotation_t_idx[['x','y','z']].values * rescale
    coords = np.round(coords_orig).astype(int)
    seg_ID = seg[coords[:,2],coords[:,1],coords[:,0]]
    unique_values, inverse_indices, counts = np.unique(seg_ID[seg_ID>0], return_inverse=True, return_counts=True)
    unique_seg_ID = unique_values[counts==1].tolist()
    removed_seg_ID = np.array(list(set(np.arange(1,np.max(seg)+1)) - set(unique_seg_ID))) - 1
    return removed_seg_ID







def model_pair_neurons_zephir_frame(path_graph,t1,t2,removed_seg_ID,selected_index,model_all,img_shape,threshold,nearby_search_num_list,with_AM,device,train_val):
    '''
    Getting the merged neuron index from the newest 
    '''
    
    
    
    #### The following module is to produce the pandas dataframe       
    data1, data2 = get_pair_data4(t1,t2,device,path_graph)
    
    data1.y = np.ones(len(data1.x))
    data1.y[removed_seg_ID] = 0
    
    
    ### To obatain the predicted class in the data2 and built the partial annotations at frame t2
    edge_label_matrix = torch.ones(len(data1.x),len(data2.x)).to(device)
    for nearby_search_num in nearby_search_num_list: 
        edge_label_matrix_model = switch_prediction(model_all[nearby_search_num],data1,data2,threshold,nearby_search_num,with_AM,device,train_val)
        edge_label_matrix = edge_label_matrix * edge_label_matrix_model

    #### Getting the class in data1 from the annotations at frame t1 and the segmentation
    edge_label_matrix[data1.y==0,:] = 0 ## For the reference frame, if the neuron is not uniquely segmented, then ignore
    AM_mask = get_AM_mask(data1,data2,5,with_AM,device,train_val)

    
    AM_mask[data1.y==0] = False
    
    ### Get the predicted class in the data2 from the predicted matrix
    ind = torch.where((edge_label_matrix*(AM_mask > 0))>0)
    # data2_y = torch.zeros(len(data2.x)).to(torch.int).to(device)
    # data2_y[ind[1]] = (data1.y[ind[0]]).to(torch.int)
    selected_index = torch.tensor([8,7,2],dtype = torch.long).to(device) ## ## the coordinates index
    # coords_2 = data2.x[selected_index]

    ind = torch.where((edge_label_matrix*AM_mask)>0)
    norm_factor = (torch.tensor(img_shape) - 1).to(device)
    parent_coords = (data1.x[ind[0]][:,selected_index]/norm_factor).cpu()
    child_coords = (data2.x[ind[1]][:,selected_index]/norm_factor).cpu()

    if train_val == 'train':
        df_t2 = {
            't_idx': t2,
            'x': child_coords[:,0],
            'y': child_coords[:,1],
            'z': child_coords[:,2],
            'parent_id': t1,
            'parent_coords_x': parent_coords[:,0],
            'parent_coords_y': parent_coords[:,1],
            'parent_coords_z': parent_coords[:,2],
            'worlidline_id':(data1.y[ind[0]]-1).cpu().to(torch.int)
             }

    elif train_val == 'eval':
        df_t2 = {
            't_idx': t2,
            'x': child_coords[:,0],
            'y': child_coords[:,1],
            'z': child_coords[:,2],
            'parent_id': t1,
            'parent_coords_x': parent_coords[:,0],
            'parent_coords_y': parent_coords[:,1],
            'parent_coords_z': parent_coords[:,2],
             }

    df = pd.DataFrame.from_dict(df_t2)
    return df





def matched_df(df,annotation,t1,t2,rescale):
    '''
    filter by the distance
    return the keep index and filterd df
    '''
    annotation_t_idx = annotation[annotation['t_idx']==t1]
    coords_t1 = annotation_t_idx[['y','x','z']].values * rescale
    coords_t1_seg =  df[['parent_coords_y','parent_coords_x','parent_coords_z']].values * rescale
    
    # dist_matrix  = distance.cdist(coords_t1_seg,coords_t1)
    # ind = np.argmin(dist_matrix,axis = 1)
    # df['worldline_id'] = annotation_t_idx['worldline_id'].values[ind]
    # df['dist'] = np.min(dist_matrix,axis = 1)
    # df = df.sort_values(by=['worldline_id'])
    # df_filtered = df[df['dist']<=6 ]
    
    dist_matrix  = distance.cdist(coords_t1_seg,coords_t1)
    
    dist_xy_matrix  = distance.cdist(coords_t1_seg[:,0:2],coords_t1[:,0:2])
    dist_x_matrix  = distance.cdist(coords_t1_seg[:,0:1],coords_t1[:,0:1])
    dist_y_matrix  = distance.cdist(coords_t1_seg[:,1:2],coords_t1[:,1:2])
    dist_z_matrix  = distance.cdist(coords_t1_seg[:,2:3],coords_t1[:,2:3])
    ind = np.argmin(dist_matrix,axis = 1)
    unique_elements, counts = np.unique(ind, return_counts=True)
    is_unique = np.array([counts[unique_elements == x] == 1 for x in ind]).flatten()

    df['worldline_id'] = annotation_t_idx['worldline_id'].values[ind]
    df['dist'] = np.min(dist_matrix,axis = 1)
    df['ind_unique'] = is_unique 
    df['dist_xy'] = dist_xy_matrix[np.arange(len(dist_xy_matrix)),ind]
    df['dist_x'] = dist_x_matrix[np.arange(len(dist_x_matrix)),ind]
    df['dist_y'] = dist_y_matrix[np.arange(len(dist_y_matrix)),ind]
    df['dist_z'] = dist_z_matrix[np.arange(len(dist_z_matrix)),ind]
    
    df = df.sort_values(by=['worldline_id'])
    df_filtered = df[(df['dist']<=4) & (df['ind_unique']==True) & (df['dist_x']<=2.5) & (df['dist_y']<=2.5) & (df['dist_z']<=1.5) & (df['dist_xy']<=4)]

    
    
    matched_id = df_filtered['worldline_id'].values
    orig_id = annotation[annotation['t_idx']==t2]['worldline_id'].values.tolist()
    index = np.array([orig_id.index(Id) for Id in matched_id ], dtype=int)
    return index, df_filtered

   

    