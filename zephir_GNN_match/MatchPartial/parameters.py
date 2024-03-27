

### parameters for graph generation
# dataset_path = '/home/hangdeng/work/GNN_matching_git/code/02_GNN_match/ZM9624/'
dataset_path = '/work/venkatachalamlab/Hang/GNN_matching_git/code/GNN_match/ZM9624/'
img_h5_path = dataset_path + '/data.h5'
seg_h5_path = '/work/venkatachalamlab/Hang/Matching_neuron/seg/'
annotation_path = dataset_path +'/annotations_test.h5' ## initially, zephir will create an empty list of the annotation, here I used the empty annotations.h5
t_max = 1060 ## the maximum time index in this video




### parameters for GNN matching
nearby_search_num_list = [5,20]
device = 'cuda' ## 'cuda' or 'cpu'
with_AM = False  ## Please use defaule False
train_val = 'eval' ## Use 'eval' for prediction; 'train' for training the models which contatain the ground truth from the annotaiton's worldline
# model_path = '/home/hangdeng/work/GNN_matching_git/code/02_GNN_match/model_weights/loss_train.pt' ## the path of trained model weights
model_path = '/work/venkatachalamlab/Hang/GNN_matching_git/code/GNN_match/model_weights/loss_train.pt'
threshold = 0.9 ## The threshold of the probablity to define whether two neurons matches
img_shape = [512,512,23] ## the 3D stack of images size
save_matched_path = '/home/hangdeng/work/GNN_matching_git/code/02_GNN_match/annotations_partial.h5' ## the saving path name of the matched partial annotation.h5
nearby_search_num = [5,20] ## list of nearby neurons that is taken into consideration
path_graph = '/work/venkatachalamlab/Hang/Matching_neuron/graph/'## the saving path of graph for each time index, the graph path contain nodes and edge features for each frame












