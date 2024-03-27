import time
import shutil
import sys
from MatchPartial.parameters import *
from MatchPartial.Graph_data import *
from frame_order_zephir import *
from MatchPartial.eval_func import *



#### if using slurm in the run
# def read_args():
#     use_GNN = sys.argv[1] == 'True'
#     file_name = sys.argv[2]
#     t_initil_list = sys.argv[3]
#     return use_GNN, file_name, t_initil_list

# use_GNN, file_name,t_initil_list = read_args()



use_GNN, file_name,t_initil_list = True, "annotations.h5", [444]
container.update({'t_annot':t_initil_list})
print("use_GNN",use_GNN)
print("file_name",file_name)
print("t_initil_list",t_initil_list)

  
# pathlib.Path('graph/').mkdir(parents=True, exist_ok=True) 



#### Extract the features from the graph
def save_data(t_idx):
    [ch,num_nearest] = [1,5] ## ch is which channel to load the image, for NeuroPal please convert into RGB then use it
    data = get_data_generator(t_idx, ch, num_nearest,img_h5_path,seg_h5_path,annotation_path,train_val)
    torch.save(data, f'{path_graph}{t_idx}.pt')

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(save_data, tqdm(range(0,t_max )))


#### Match partial annotations from the features and GNN model
path_graph = 'graph/' 
nearby_search_num_list = [5]
model_all = load_model_args(model_path, nearby_search_num_list,with_AM, device,train_val)
df_all = model_match_patial_neurons(path_graph,t_frame,model_all,img_shape,threshold,nearby_search_num_list,with_AM,device,train_val)
save_pandas_h5(save_matched_path, df_all)
print("Output all the matched annotaiton in the path: ", save_matched_path)

print("The time of running the whole segmented datasets: ",t1-t0)
print("the number of found partial annotations: ",len(df_all)/(np.max(t_frame)+1))




t0 = time.time()

rescale = np.array(img_shape) - 1

seg_path = '/work/venkatachalamlab/Hang/Matching_neuron/seg/'
nearby_search_num_list = [5]
selected_index = torch.tensor([8,7,2],dtype = torch.long)
model_all = load_model_args(model_path, nearby_search_num_list,with_AM, device,train_val)
# annotation_orig = get_annotation_df(dataset)
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


print("starting")
t_list = container.get('t_list')
p_list = container.get('p_list')
t_ref = []

for t2 in t_list:

    t1 = p_list[t2]


    if use_GNN == True:
        ### get the annotation at time t2 from GNN match
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


    print("before expected parent id",t1)
    print("before the parent id",(np.unique(annotation[annotation['t_idx']==t2]['parent_id'].values)))
    

    t_ref.append(t1)
    t_ref.append(t2)
    t_ref = np.unique(np.array(t_ref)).tolist()
    container.update({'t_annot':t_ref})
    print("---before loading",np.sum(results[t2]))


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
    
    ### save the tracking results into dataset
    
   
    
    save_annotations(
    container=container,
    results=results,
    save_mode='o')
    annotation = get_annotation_df(dataset)
    

    print("-----------expected parent id",t1)
    print("-----------the parent id",(np.unique(annotation[annotation['t_idx']==t2]['parent_id'].values)))
    

    
# src = '/work/venkatachalamlab/Hang/GNN_matching_git/code/02_GNN_match/ZM9624/annotations.h5'
# shutil.copyfile(src, dst)
