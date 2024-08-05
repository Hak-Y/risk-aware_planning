
import yaml
import math
import numpy as np

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def load_config(path, default_path=None):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f,Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f,Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def define_align_matrix(align_config):
    ''' Get transformation matrix to align scene

        Args:
            align_config : aligning configuration
        '''
    px=align_config['px']
    py=align_config['py']
    pz=align_config['pz']
    qx=align_config['qx']
    qy=align_config['qy']
    qz=align_config['qz']
    qw=align_config['qw']
    # Convert quaternion to Euler angles
    deg_x = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    deg_y = math.asin(2 * (qw * qy - qz * qx))
    deg_z = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    cos_theta, sin_theta = math.cos(math.radians(deg_x)), math.sin(math.radians(deg_x))
    cos_alpha, sin_alpha = math.cos(math.radians(deg_y)), math.sin(math.radians(deg_y))
    cos_beta, sin_beta = math.cos(math.radians(deg_z)), math.sin(math.radians(deg_z))

    rot_1 = np.array([[1, 0, 0, 0],
                   [0, cos_theta, -sin_theta, 0],
                   [0, sin_theta, cos_theta, 0],
                   [0, 0, 0, 1]])

    rot_2 = np.array([[cos_alpha, 0, sin_alpha, 0],
                   [0, 1, 0, 0],
                   [-sin_alpha, 0, cos_alpha, 0],
                   [0, 0, 0, 1]])

    rot_3 = np.array([[cos_beta, -sin_beta, 0, 0],
                   [sin_beta, cos_beta, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    shift_mat = np.array([[1, 0, 0, px],
                          [0, 1, 0, py],
                          [0, 0, 1, pz],
                          [0, 0, 0, 1]])

    align_matrix = rot_1

    align_matrix = np.matmul(rot_2, align_matrix)
    align_matrix = np.matmul(rot_3, align_matrix)
    align_matrix = np.matmul(shift_mat, align_matrix)

    return align_matrix
