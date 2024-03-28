import time
import shutil
import sys
from MatchPartial.eval_func import *
from MatchPartial.Graph_data import *
from Segmentation.Video_seg import *
from frame_order_zephir import *
from parameters_begin import *





container.update({'t_annot':t_initil_list})
print("use_GNN",use_GNN)
print("file_name",file_name)
print("t_initil_list",t_initil_list)




##########################################################################################################################
##                           module for segmentation and extract feature into graph                                     ##
##########################################################################################################################
model_3D, model_2D = load_model_3D_and_2D(seg_model_weights_path)
paths = [seg_h5_path, path_graph]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
img_shape = get_img_shape(dataset_path)
for t_idx in range(img_shape[0]):
    label_z = get_frame_segmentation(model_3D,model_2D,t_idx, ch, zoom_factor,dataset_path)
    save_var_h5(os.path.join(seg_h5_path ,str(t_idx)+'.h5'),[label_z],['label'])
    num_nearest = 5 ## create the edge index in the graph by taking the nearby 5 neurons into consideration
    data = get_data_generator(t_idx, ch, num_nearest,img_h5_path,seg_h5_path,annotation_path,train_val)
    torch.save(data, f'{path_graph}/{t_idx}.pt')
print("finish the segmentation of movie in the folder path: ", seg_h5_path)    
###########################################################################################################################
    




##########################################################################################################################
##                             initialization the annotation for zephir tracking                                        ##
##########################################################################################################################
t0 = time.time()
rescale = np.array(img_shape) - 1
nearby_search_num_list = [5]
selected_index = torch.tensor([8,7,2],dtype = torch.long)
model_all = load_model_args(model_path, nearby_search_num_list,with_AM, device,train_val)
annotation_orig = get_annotation_file_df(dataset, "annotations_orig.h5") 
annotation = copy.deepcopy(annotation_orig)
annotation[['x','y','z']] = 0.5
for t_initial in t_initil_list:
    annotation[annotation['t_idx']==t_initial] = annotation_orig[annotation_orig['t_idx']==t_initial] 
save_pandas_h5(dataset_path+file_name, annotation)




### initialize the tracking results and annotations
shape_t = container.get('shape_t')
shape_n = container.get('shape_n')
results = np.zeros((shape_t, shape_n, 3))
t_ref=eval(args['--t_ref']) if args['--t_ref'] else None,
wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
n_ref=int(args['--n_ref']) if args['--n_ref'] else None,
print("starting tracking")
t_list = container.get('t_list')
p_list = container.get('p_list')
t_ref = []


for t2 in t_list:
    t1 = p_list[t2]

    if use_GNN == True:
        ### get the matched partial annotation at time t2 from GNN match
        removed_seg_ID = find_merged_index_graphdata(seg_path,t1,img_shape,annotation)
        df = model_pair_neurons_zephir_frame(path_graph,t1,t2,removed_seg_ID,selected_index,
                                             model_all,img_shape,threshold,nearby_search_num_list,with_AM,device,train_val)
        index, df_filtered = matched_df(df,annotation,t1,t2,rescale)

        print("originalally found matches:", len(df), "after filtered the merged neurons, found matches: ",len(df_filtered))

        ## only take the unique worldline    
        indices = annotation[annotation['t_idx'] == t2].index
        annotation.loc[indices[index], ['x', 'y', 'z']] = df_filtered[['x', 'y', 'z']].values
        annotation.loc[indices[index], ['provenance']] = b'GNN'
        annotation.loc[indices[index], ['parent_id']] = t1
        annotation.loc[indices[index], ['t_idx']] = t2
        print("the updated annotation at time t1",np.sum(abs(annotation[annotation['t_idx'] == t1][['x','y','z']].values-0.5)))
        print("the updated annotation at time t2",np.sum(abs(annotation[annotation['t_idx'] == t2][['x','y','z']].values-0.5)))

        # container.update({'partial_annot' :annotation.loc[indices[index]]})
    print("before expected parent id",t1)
    print("before the parent id",(np.unique(annotation[annotation['t_idx']==t2]['parent_id'].values)))
    

    t_ref.append(t1)
    t_ref.append(t2)
    t_ref = np.unique(np.array(t_ref)).tolist()
    container.update({'t_annot':t_ref})
    print("---before loading",np.sum(results[t2]))
    
    # container, results = build_partial_annotations(container,annotation,results,file_name,t_ref,None,None,)
    container, results = build_partial_annotations_test(container,annotation,results,t_ref,None,None,)
    
    
    
    save_annotations_filename(
    container=container,
    results=results,
    annotation=annotation,
    filename=file_name,
    save_mode='o')
    
    
    print("---after loading",np.sum(results[t2]))
    container, results = track_all(
        container=container,
        results=results,
        zephir=zephir,
        zephod=zephod,
        clip_grad=float(args['--clip_grad']),
        lambda_t=float(args['--lambda_t']),
        lambda_d=float(args['--lambda_d']),
        lambda_n=float(args['--lambda_n']),
        lambda_n_mode='norm',
        lr_ceiling=float(args['--lr_ceiling']),
        # lr_coef=2.0,
        lr_floor=float(args['--lr_floor']),
        motion_predict=bool(args['--motion_predict']),
        n_epoch=int(args['--n_epoch']),
        n_epoch_d=int(args['--n_epoch_d']),
        _t_list= np.array([t2]),
    )
    print("---!!time",t2, results[t2][0])
    

    
    save_annotations(
    container=container,
    results=results,
    save_mode='o')
    annotation = get_annotation_df(dataset)
    
    

    print("-----------expected parent id",t1)
    print("-----------the parent id",(np.unique(annotation[annotation['t_idx']==t2]['parent_id'].values)))
    

    
# src = '/work/venkatachalamlab/Hang/GNN_matching_git/code/02_GNN_match/ZM9624/annotations.h5'
# shutil.copyfile(src, dst)
