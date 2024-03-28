################################################################################################
## This module is to get the frame order that zephir is running
## specify the dataset folder: dataset_path
## specify the arguments used for zephir: args
## output the order of the tracking frame as list :t_frame
################################################################################################
# from zephir.methods import *
# from zephir.models.container import Container
# from zephir.utils.io import *

# from zephir.methods.recommend_frames import recommend_frames
# from zephir.methods.extract_traces import extract_traces

# # %matplotlib inline
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from MatchPartial.parameters import *
# import copy


# dataset = Path(dataset_path)
# dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# # args_name = 'args-Copy1.json'
# # with open(str(dataset / args_name)) as json_file:
# #     args = json.load(json_file)
# # args['--sort_mode'] = 'similarity'



# #### The flowing arguments confirmed by James for tracking ZM9624
# args = {'--dimmer_ratio': '0.1',
#  '--n_epoch_d': '0',
#  '--motion_predict': 'True',
#  '--grid_shape': '49',
#  '--fovea_sigma': '10.0',
#  '--lr_coef': '2.0',
#  '--t_track': None,
#  '--allow_rotation': 'True',
#  '--z_compensator': '4.0',
#  '--n_epoch': '40',
#  '--channel': '1',
#  '--include_all': 'False',
#  '--save_mode': 'w',
#  '--gamma': '2',
#  '--lambda_n_mode': 'norm',
#  '--exclusive_prov': None,
#  '--lr_floor': '0.01',
#  '--load_checkpoint': 'False',
#  '--lambda_t': '-1.0',
#  '--load_args': 'True',
#  '--wlid_ref': None,
#  '--cuda': 'True',
#  '--nn_max': '5',
#  '--sort_mode': 'similarity',
#  '--n_chunks': '1',
#  '--lambda_n': '0.1',
#  '--lambda_d': '0.1',
#  '--clip_grad': '-1',
#  '--n_frame': '1',
#  '--t_ignore': None,
#  # '--t_ref': '498,444,463',
#  '--t_ref': '444',
#  '--load_nn': 'False',
#  '--n_ref': None,
#  '--lr_ceiling': '0.1',
#  '--exclude_self': 'True'}


# container = Container(
#     dataset = dataset,
#     allow_rotation=args['--allow_rotation'] in ['True', 'Y', 'y'],
#     channel=int(args['--channel']) if args['--channel'] else None,
#     dev=dev,
#     exclude_self=args['--exclude_self'] in ['True', 'Y', 'y'],
#     exclusive_prov=(bytes(args['--exclusive_prov'], 'utf-8')
#                     if args['--exclusive_prov'] else None),
#     gamma=float(args['--gamma']),
#     include_all=args['--include_all'] in ['True', 'Y', 'y'],
#     lr_coef=float(args['--lr_coef']),
#     n_frame=int(args['--n_frame']),
#     z_compensator=float(args['--z_compensator']),
# )

from zephir.methods import *
from zephir.models.container import Container
from zephir.utils.io import *

from zephir.methods.recommend_frames import recommend_frames
from zephir.methods.extract_traces import extract_traces

# %matplotlib inline
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from parameters import *
import copy


dataset = Path(dataset_path)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# args_name = 'args-Copy1.json'
# with open(str(dataset / args_name)) as json_file:
#     args = json.load(json_file)
# args['--sort_mode'] = 'similarity'



#### The flowing arguments confirmed by James for tracking ZM9624



container = Container(
    dataset = dataset,
    allow_rotation=args['--allow_rotation'] in ['True', 'Y', 'y'],
    channel=int(args['--channel']) if args['--channel'] else None,
    dev=dev,
    exclude_self=args['--exclude_self'] in ['True', 'Y', 'y'],
    exclusive_prov=(bytes(args['--exclusive_prov'], 'utf-8')
                    if args['--exclusive_prov'] else None),
    gamma=float(args['--gamma']),
    include_all=args['--include_all'] in ['True', 'Y', 'y'],
    lr_coef=float(args['--lr_coef']),
    n_frame=int(args['--n_frame']),
    z_compensator=float(args['--z_compensator']),
)



container, results = build_annotations(
    container=container,
    annotation=None,
    t_ref=eval(args['--t_ref']) if args['--t_ref'] else None,
    # t_ref=None,
    wlid_ref=eval(args['--wlid_ref']) if args['--wlid_ref'] else None,
    n_ref=int(args['--n_ref']) if args['--n_ref'] else None,
)

container, zephir, zephod = build_models(
    container=container,
    dimmer_ratio=float(args['--dimmer_ratio']),
    grid_shape=(5, 2 * (int(args['--grid_shape']) // 2) + 1,
                2 * (int(args['--grid_shape']) // 2) + 1),
    fovea_sigma=(1, float(args['--fovea_sigma']),
                 float(args['--fovea_sigma'])),
    n_chunks=int(args['--n_chunks']),
)

container = build_springs(
    container=container,
    load_nn=args['--load_nn'] in ['True', 'Y', 'y'],
    nn_max=int(args['--nn_max']),
)

container = build_tree(
    container=container,
    sort_mode=str(args['--sort_mode']),
#     t_ignore=eval(args['--t_ignore']) if args['--t_ignore'] else None,
    t_ignore=None,
#     t_track=eval(args['--t_track']) if args['--t_track'] else None,
    t_track=None,
)


t_frame = list(container.get('t_annot')) +list(container.get('t_list'))

### container.get('p_list') : parent frame list
### p_list[t] for t in t_list, t is the child frame
####################################The following module is to build the function to update the partial annotations#########
import shutil
import pandas as pd
from docopt import docopt
from zephir.utils.utils import *


def build_partial_annotations(
    container,
    annotation,
    results,
    t_ref,
    wlid_ref,
    n_ref,):
    """Load and handle annotations from annotations.h5 file.

    Annotations are loaded and sorted according to user arguments, and used to
    populate an empty results array. This array will be filled during tracking.

    :param container: variable container; needs to contain: dataset,
    exclude_self, exclusive_prov, shape_t
    :param annotation: override annotations to use instead of loading from file
    :param t_ref: override frames to use as annotations
    :param wlid_ref: override worldline id's to analyze
    :param n_ref: override maximum number of keypoints to analyze
    :return: container (updated entries for: annot, shape_n, partial_annot,
    provenance, t_annot, worldline_id), results (pre-filled with loaded annotations)
    """

    # pull variables from container
    dataset = container.get('dataset')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    shape_t = container.get('shape_t')

    # checking annotated frames
    if annotation is None:
        # annotation = get_annotation_df(dataset)
        annotation = get_annotation_file_df(dataset, file_name)
        
    t_annot = np.unique(annotation['t_idx']).astype(int)
    if t_ref is not None:
        if type(t_ref) is int:
            t_ref = [t_ref]
        t_annot = np.array([t for t in t_ref if t in t_annot]).astype(int)
    t_annot = np.sort(t_annot)

    worldline_id = None
    if wlid_ref is not None:
        if type(wlid_ref) is int:
            worldline_id = np.arange(wlid_ref)
        elif type(wlid_ref) is tuple and len(wlid_ref) == 2:
            worldline_id = np.arange(min(wlid_ref), max(wlid_ref))
        elif type(wlid_ref) is tuple or type(wlid_ref) is list:
            worldline_id = np.sort(np.array(wlid_ref))
        shape_n = len(worldline_id)
    elif n_ref is not None:
        shape_n = n_ref
        for t in t_annot:
            u, _, _ = get_annotation(annotation, t, exclusive_prov, exclude_self)
            if len(u) == n_ref:
                worldline_id = u
                print(f'Using frame #{t} as initial reference with specified {n_ref} annotations...')
                break
    else:
        nn_list = [len(get_annotation(annotation, t, exclusive_prov, exclude_self)[0]) for t in t_annot]
        shape_n, t_max = np.max(nn_list), t_annot[np.argmax(nn_list)]
        worldline_id, _, _ = get_annotation(annotation, t_max, exclusive_prov, exclude_self)
        print(f'Using frame #{t_max} as initial reference with {shape_n} annotations found...')

    if shape_n == 0 or shape_n is None or worldline_id is None:
        print('\n******* ERROR: annotations could not be loaded properly! '
              'Check parameters: t_ref, wlid_ref, n_ref.\n\n')

    annot = []
    partial_annot = {}
    # results = np.zeros((shape_t, shape_n, 3))
    provenance = np.array([[b'ZEIR'] * shape_n] * shape_t)
    print("t_annot:",t_annot)
    for t in t_annot:
        # loading and sorting annotation by worldline_id
        u, _annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)

        # checking if worldlines are available in the annotation
        w_idx = np.array([np.where(worldline_id == w)[0][-1]
                          for w in u if w in worldline_id], dtype=int)
        u_idx = np.array([np.where(u == w)[0][-1]
                          for w in worldline_id if w in u], dtype=int)
        _annot = _annot[u_idx, ...]

        if _annot.shape[0] > shape_n or _annot.shape[0] == 0:
            t_annot = np.setdiff1d(t_annot, [t])
            continue
        elif _annot.shape[0] < shape_n:
            t_annot = np.setdiff1d(t_annot, [t])
            partial_annot[t] = (w_idx, _annot)
            results[t, w_idx] = _annot
            provenance[t, w_idx] = prov[u_idx]
            continue
        
        annot.append(_annot)
        # print("for the time index",t,len(_annot))
        results[t] = _annot
        provenance[t, w_idx] = prov[u_idx]

    print(f'\nAnnotations loaded for frames {list(t_annot)} '
          f'with shape: {np.array(annot).shape}')
    if len(partial_annot) > 0:
        print(f'*** Partial annotations found for {len(partial_annot)} frames')

    # push variables to container
    container.update({
        'annot': annot,
        'shape_n': shape_n,
        'partial_annot': partial_annot,
        'provenance': provenance,
        't_annot': t_annot,
        'worldline_id': worldline_id,
    })

    # push results to checkpoint
    update_checkpoint(dataset, {
        'results': results,
    })

    return container, results






def save_annotations_filename(
    container,
    results,
    annotation,
    filename,
    save_mode):
    """Save results to file.

    Handles tracking results and compiles to an annotations data frame to save
    to file according to save_mode: 'o' will overwrite existing annotations.h5,
    'w' will write to coordinates.h5. Existing annotations will overwrite
    ZephIR results if include_all is True.

    :param container: variable container, needs to contain: dataset, exclude_self,
    exclusive_prov, include_all, p_list, provenance, shape_t, shape_n, worldline_id
    :param results: tracking results
    :param save_mode: mode for writing to file
    """

    # pull variables from container
    dataset = container.get('dataset')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    include_all = container.get('include_all')
    p_list = container.get('p_list')
    provenance = container.get('provenance')
    shape_t = container.get('shape_t')
    shape_n = container.get('shape_n')
    worldline_id = container.get('worldline_id')

    # annotation = get_annotation_df(dataset)
    # annotation = get_annotation_file_df(dataset, filename)
    
    # saving result to .h5
    print('\nCompiling and saving results to file...')
    p_list = np.array(p_list)
    p_list[np.where(p_list==-1)] = np.where(p_list==-1)
    xyz_pd = np.concatenate(
        (np.repeat(np.arange(shape_t), shape_n)[:, np.newaxis],
         results.reshape((-1, 3)) / 2.0 + 0.5,
         np.tile(worldline_id, shape_t)[:, np.newaxis],
         np.repeat(p_list, shape_n)[:, np.newaxis],
         provenance.reshape((-1, 1))),
        axis=-1
    )

    if include_all:
        for t in np.unique(annotation['t_idx']):
            u, annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)
            w_idx = np.where(
                np.logical_and(
                    xyz_pd[t * shape_n:(t+1) * shape_n, 0].astype(np.float32) == t,
                    np.isin(xyz_pd[t * shape_n:(t+1) * shape_n, 4].astype(np.float32), u))
            )[0] + t * shape_n
            if len(w_idx) > 0:
                u_idx = np.where(np.isin(u, xyz_pd[w_idx, 4].astype(np.float32)))[0]
                xyz_pd[w_idx, :] = np.concatenate(
                    (np.ones((len(u_idx), 1)) * t,
                     annot[u_idx, ...] / 2 + 0.5,
                     u[u_idx, np.newaxis],
                     np.ones((len(u_idx), 1)) * p_list[t],
                     prov[u_idx, np.newaxis]),
                    axis=-1
                )
            if exclusive_prov is not None or exclude_self is True:
                u, annot, prov = get_annotation(annotation, t, None, False)
                w_idx = np.where(
                    np.logical_and(
                        xyz_pd[t * shape_n:(t + 1) * shape_n, 0].astype(np.float32) == t,
                        np.isin(xyz_pd[t * shape_n:(t + 1) * shape_n, 4].astype(np.float32), u))
                )[0] + t * shape_n
            if len(u) > len(w_idx):
                u_idx = np.where(np.isin(u, xyz_pd[w_idx, 4].astype(np.float32), invert=True))[0]
                xyz_pd = np.append(
                    xyz_pd,
                    np.concatenate(
                        (np.ones((len(u_idx), 1)) * t,
                         annot[u_idx, ...] / 2 + 0.5,
                         u[u_idx, np.newaxis],
                         np.ones((len(u_idx), 1)) * p_list[t],
                         prov[u_idx, np.newaxis]),
                        axis=-1
                    ),
                    axis=0
                )

    columns = {
        't_idx': np.uint32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'worldline_id': np.uint32,
        'parent_id': np.uint32,
        'provenance': np.dtype("S4"),
    }
    if save_mode == 'o':
        if not (dataset / 'backup').is_dir():
            Path.mkdir(dataset / 'backup')
        now = datetime.datetime.now()
        now_ = now.strftime("%Y_%m_%d_%H_%M_%S")
        shutil.copy(dataset / filename,
                    dataset / 'backup' / f'annotations_{now_}.h5')
        f = h5py.File(dataset / filename, mode='w')
    else:
        f = h5py.File(dataset / 'coordinates.h5', mode=save_mode)

    data = np.array(list(range(1, xyz_pd.shape[0] + 1)), dtype=np.uint32)
    f.create_dataset('id', shape=(xyz_pd.shape[0], ), dtype=np.uint32, data=data)

    for i, c in enumerate(columns.keys()):
        if c == 'provenance':
            data = np.array(xyz_pd[:, i], dtype=columns[c])
        else:
            data = np.array(xyz_pd[:, i].astype(np.float32), dtype=columns[c])
        f.create_dataset(c, shape=(xyz_pd.shape[0], ), dtype=columns[c], data=data)

    f.close()

    return




def get_annotation_file_df(dataset: Path, file_name: str) -> pd.DataFrame:
    """Load and return annotations as an ordered pandas dataframe.
    This should contain the following:
    - t_idx: time index of each annotation
    - x: x-coordinate as a float between (0, 1)
    - y: y-coordinate as a float between (0, 1)
    - z: z-coordinate as a float between (0, 1)
    - worldline_id: track or worldline ID as an integer
    - provenance: scorer or creator of the annotation as a byte string
    """
    print("")
    with h5py.File(dataset / file_name, 'r') as f:
        data = pd.DataFrame()
        for k in f:
            data[k] = f[k]
    return data




def build_partial_annotations_test(container,annotation,results,t_ref,wlid_ref,n_ref):
    print("starting")
    # pull variables from container
    dataset = container.get('dataset')
    exclude_self = container.get('exclude_self')
    exclusive_prov = container.get('exclusive_prov')
    shape_t = container.get('shape_t')

    # checking annotated frames
    # if annotation is None:
    #     # annotation = get_annotation_df(dataset)
    #     annotation = get_annotation_file_df(dataset, file_name)
        
    t_annot = np.unique(annotation['t_idx']).astype(int)
    if t_ref is not None:
        if type(t_ref) is int:
            t_ref = [t_ref]
        t_annot = np.array([t for t in t_ref if t in t_annot]).astype(int)
    t_annot = np.sort(t_annot)

    worldline_id = None
    if wlid_ref is not None:
        if type(wlid_ref) is int:
            worldline_id = np.arange(wlid_ref)
        elif type(wlid_ref) is tuple and len(wlid_ref) == 2:
            worldline_id = np.arange(min(wlid_ref), max(wlid_ref))
        elif type(wlid_ref) is tuple or type(wlid_ref) is list:
            worldline_id = np.sort(np.array(wlid_ref))
        shape_n = len(worldline_id)
    elif n_ref is not None:
        
        shape_n = n_ref
        for t in t_annot: 
            u, _, _ = get_annotation(annotation, t, exclusive_prov, exclude_self)
            if len(u) == n_ref:
                worldline_id = u
                print(f'Using frame #{t} as initial reference with specified {n_ref} annotations...')
                break
    else:
        nn_list = [len(get_annotation(annotation, t, exclusive_prov, exclude_self)[0]) for t in t_annot]
        shape_n, t_max = np.max(nn_list), t_annot[np.argmax(nn_list)]
        worldline_id, _, _ = get_annotation(annotation, t_max, exclusive_prov, exclude_self)
        print(f'Using frame #{t_max} as initial reference with {shape_n} annotations found...')

    if shape_n == 0 or shape_n is None or worldline_id is None:
        print('\n******* ERROR: annotations could not be loaded properly! '
              'Check parameters: t_ref, wlid_ref, n_ref.\n\n')

    annot = []
    
    partial_annot = {}
    # results = np.zeros((shape_t, shape_n, 3))
    
    provenance = np.array([[b'ZEIR'] * shape_n] * shape_t)
    print("t_annot:",t_annot)
    for t in t_annot:
        # loading and sorting annotation by worldline_id
        u, _annot, prov = get_annotation(annotation, t, exclusive_prov, exclude_self)
        # print("debugging",t,_annot.shape)
        # checking if worldlines are available in the annotation
        w_idx = np.array([np.where(worldline_id == w)[0][-1]
                          for w in u if w in worldline_id], dtype=int)
        u_idx = np.array([np.where(u == w)[0][-1]
                          for w in worldline_id if w in u], dtype=int)
        _annot = _annot[u_idx, ...]

        if _annot.shape[0] > shape_n or _annot.shape[0] == 0:
            t_annot = np.setdiff1d(t_annot, [t])
            continue
        elif _annot.shape[0] < shape_n:
            t_annot = np.setdiff1d(t_annot, [t])
            partial_annot[t] = (w_idx, _annot) ### index of the worldline id
            results[t, w_idx] = _annot
            provenance[t, w_idx] = prov[u_idx]
            print(f'*** Partial annotations found for {len(_annot)} frames',"at time:", t)
            continue
        
        annot.append(_annot)
        print("for the time index",t,len(_annot))
        results[t] = _annot
        provenance[t, w_idx] = prov[u_idx]

    print(f'\nAnnotations loaded for frames {list(t_annot)} '
          f'with shape: {np.array(annot).shape}')
    if len(partial_annot) > 0:
        print(f'*** Partial annotations found for {len(partial_annot)} frames')

    # push variables to container
    container.update({
        'annot': annot,
        'shape_n': shape_n,
        'partial_annot': partial_annot,
        'provenance': provenance,
        't_annot': t_annot,
        'worldline_id': worldline_id,
    })

    # push results to checkpoint
    update_checkpoint(dataset, {
        'results': results,
    })

    return container, results





