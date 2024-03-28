import os
import json


### parameters for the dataset
dataset_path = '/work/venkatachalamlab/Hang/GNN_matching_git/dataset/GNN_match/ZM9624/'
# dataset_path = '/work/venkatachalamlab/Hang/GNN_matching_git/dataset/static_fluid_20220327_4_herm'
metadata_path = os.path.join(dataset_path,'metadata.json')
f = open(metadata_path,) 
metadata = json.load(f) 

img_h5_path = dataset_path + '/data.h5'
seg_h5_path = os.path.join(dataset_path, 'seg')
seg_path = seg_h5_path
path_graph = os.path.join(dataset_path, 'graph')## the saving path of graph for each time index, the graph path contain nodes and edge features for each frame
annotation_path = dataset_path +'/annotations_test.h5' ## initially, zephir will create an empty list of the annotation, here I used the empty annotations.h5
t_max = metadata['shape_t'] ## the maximum time index in this video
# t_max = 20



### parameters for Segmentation and graph converting
current_folder = os.getcwd()
seg_model_weights_path = os.path.join(current_folder,'Segmentation/model_weights/weights_best_42stacks_all.h5')
ch = 1  ### specify the channel used for tracking
zoom_factor = 2  ### behaving worms likeZM9624 use:2, static_fluid use:1





### parameters for GNN matching
model_path = os.path.join(current_folder,'MatchPartial/model_weights/loss_train.pt')
nearby_search_num_list = [5,20]
device = 'cuda' ## 'cuda' or 'cpu'
with_AM = False  ## Please use defaule False
train_val = 'eval' ## Use 'eval' for prediction; 'train' for training the models which contatain the ground truth from the annotaiton's worldline
threshold = 0.9 ## The threshold of the probablity to define whether two neurons matches
img_shape = [metadata['shape_y'],metadata['shape_x'],metadata['shape_z']] ## the 3D stack of images size
nearby_search_num = [5,20] ## list of nearby neurons that is taken into consideration
use_GNN = True
file_name = "annotations.h5"
t_initial_list = [444] ## the initail annotated frame



args = {'--dimmer_ratio': '0.1',
 '--n_epoch_d': '0',
 '--motion_predict': 'True',
 '--grid_shape': '49',
 '--fovea_sigma': '10.0',
 '--lr_coef': '2.0',
 '--t_track': None,
 '--allow_rotation': 'True',
 '--z_compensator': '4.0',
 '--n_epoch': '40',
 '--channel': '1',
 '--include_all': 'False',
 '--save_mode': 'w',
 '--gamma': '2',
 '--lambda_n_mode': 'norm',
 '--exclusive_prov': None,
 '--lr_floor': '0.01',
 '--load_checkpoint': 'False',
 '--lambda_t': '-1.0',
 '--load_args': 'True',
 '--wlid_ref': None,
 '--cuda': 'True',
 '--nn_max': '5',
 '--sort_mode': 'similarity',
 '--n_chunks': '1',
 '--lambda_n': '0.1',
 '--lambda_d': '0.1',
 '--clip_grad': '-1',
 '--n_frame': '1',
 '--t_ignore': None,
 # '--t_ref': '498,444,463',
 '--t_ref': '444',
 '--load_nn': 'False',
 '--n_ref': None,
 '--lr_ceiling': '0.1',
 '--exclude_self': 'True'}






