import numpy as np
import glob

# modifications:
#  * removed combined_loss(), keras imports

def periodicVector(x0, x1, dimensions):
    '''
    Calculating periodic distances for the x, y, z dimensions
    '''
    for i in range(len(dimensions)):
        delta = x0[:, :, i] - x1[i]
        delta = np.where(delta > 0.5 * dimensions[i], delta - dimensions[i], delta)
        delta = np.where(delta < - (0.5 * dimensions[i]), delta + dimensions[i], delta)
        x0[:, :, i] = delta*4  # multiplier to rescale the values
    return x0


def orientationVector(x0, x1, dimensions):
    '''
    Calculating the orientation vector for a molecule
    '''
    x = np.copy(x0)
    for i in range(len(dimensions)):
        delta = x0[:, :, i] - x1[i]
        delta = np.where(delta > 0.5 * dimensions[i], delta - dimensions[i], delta)
        delta = np.where(delta < - (0.5 * dimensions[i]), delta + dimensions[i], delta)
        x[:,:,i] = delta
    return x


def periodicDistance(x0, x1, dimensions):
    com = np.copy(x1)
    com = com.reshape(1, 1, -1)
    com = np.repeat(com, x0.shape[0], axis=0)
    com = np.repeat(com, x0.shape[1], axis=1)
    delta = np.abs(x0 - com)
    delta = np.where(delta > np.multiply(0.5, dimensions), delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))


def get_com(x):
    '''
    Calculating the Center of Mass of the molecule.
    Using only the first 8 beads if the molecule is CHOL
    '''
    if x[0, 3] == 1:
        return np.mean(x[:8, :3], axis=0)
    else:
        return np.mean(x[:, :3], axis=0)


def get_com_head(x):
    '''
    Calculating the Center of Mass from head beads of the molecule.
    Using only the first 8 beads if the molecule is CHOL
    '''
    head_inds = np.argwhere(x[:, 6] == 1)
    return np.mean(x[head_inds, :3], axis=0)


def get_angles(x0, com, orientation, dimensions):
    vector = orientationVector(x0, com, dimensions)

    dot_product = np.dot(vector, orientation)
    cross_product = np.cross(vector, orientation)
    angle = np.arctan2(cross_product, dot_product)

    angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    return angle / (2*np.pi)


def get_local_files(data_tag="3k_run16"):
    '''
    Load data files from local directory
    '''
    if data_tag == '3k_run16':
        data_dir = '/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_run16_10us.35fs-DPPC.20-DIPC.60-CHOL.20.dir/'
    elif data_tag == '3k_run10':
        data_dir = '/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir/'
    elif data_tag == '3k_run32':
        data_dir = '/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir/'

    data_files = glob.glob('%s/*.npz' % data_dir)
    filelist = [d for d in data_files if 'AE' not in d]
    filelist = sorted(filelist)
    import pilot2_datasets as p2
    fields = p2.gen_data_set_dict()

    return (filelist, fields)


def get_data_arrays(f):

    data = np.load(f)

    X = data['features']
    nbrs = data['neighbors']
    resnums = data['resnums']

    return (X, nbrs, resnums)


def get_neighborhood_features(x, nbrs, num_nbrs):
    '''
    Create a neighborhood feature vetor for each molecule in the frame
    Drops x, y, z values
    Stores distance relative to the center of mass of the head beads of molecule as the first (0) feature.
    Stores angles relative to the orientation of molecule as the second (1) feature.

    Args:
    x: data of shape (num_molecules, num_beads, features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)
             The features are in the order:
             [relative_distance, relative_angle, 'CHOL', 'DPPC', 'DIPC', 'Head', 'Tail', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'BL6',
             'BL7', 'BL8', 'BL9', 'BL10', 'BL11', 'BL12']
    '''
    new_x_shape = np.array((x.shape[0], x.shape[1] * (x.shape[2]-1)))
    new_x_shape[1] *= num_nbrs+1
    x_wNbrs = np.zeros(new_x_shape)

    for i in range(len(x)):

        # get neighbors
        nb_indices = nbrs[i, :num_nbrs+1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]
        xy_feats = np.copy(temp_mols[:, :, :2])

        temp_mols = temp_mols[:, :, 1:]

        # calculate com
        com = get_com_head(x[i])

        # calculate orientation
        orientation = np.squeeze(orientationVector(x[i, 3, :2].reshape(1, 1, -1), x[i, 2, :2], [1., 1.]))

        # Calculate relative periodic distances from the com
        temp_mols[:, :, 0] = periodicDistance(xy_feats, com[0, :2], [1., 1.])

        temp_mols[:, :, 1] = get_angles(xy_feats, com[0, :2], orientation, [1., 1.])

        # For the CHOL molecules set the last 4 beads to all zero
        ind = np.argwhere(temp_mols[:, 0, 2] == 1)
        temp_mols[ind, 8:, :] = 0

        # Sort the nbrs by angle of the 1st bead
        sorted_arg = np.argsort(temp_mols[1:, 0, 1]) + 1
        temp_mols[1:, :, :] = temp_mols[sorted_arg, :, :]

        newshape = (1, np.prod(temp_mols.shape))
        temp_mols = np.reshape(temp_mols, newshape)

        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols

    return x_wNbrs


def append_nbrs_relative(x, nbrs, num_nbrs):
    '''
    Appends the neighbors to each molecule in the frame
    Also, uses x, y, z positions relative to the center of mass of the molecule.

    Args:
    x: data of shape (num_molecules, num_beads, features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)

    '''
    new_x_shape = np.array((x.shape[0], np.prod(x.shape[1:])))
    new_x_shape[1] *= num_nbrs+1
    x_wNbrs = np.zeros(new_x_shape)

    for i in range(len(x)):
        # get neighbors
        nb_indices = nbrs[i, :num_nbrs+1].astype(int)
        nb_indices = nb_indices[nb_indices != -1]
        temp_mols = x[nb_indices]

        # calculate com
        com = get_com(x[i])

        # Calculate relative periodic distances from the com
        temp_mols = periodicVector(temp_mols, com, [1., 1., 0.3])  # The absolute span of z-dimension is about third of x and y

        # For the CHOL molecules set the last 4 beads to all zero
        ind = np.argwhere(temp_mols[:, 1, 3] == 1)
        temp_mols[ind, 8:, :] = 0

        newshape = (1, np.prod(temp_mols.shape))
        temp_mols = np.reshape(temp_mols, newshape)

        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols

    return x_wNbrs


def append_nbrs(x, nbrs, num_nbrs):
    '''
    Appends the neighbors to each molecule in the frame

    Args:
    x: data of shape (num_molecules, num_beads*features)
    nbrs: neighbor index of molecules of shape (num_molecules, 100)
    num_nbrs: int, number of neighbors to append

    Returns:
    x_wNbrs: concatenated features of all neighbors of shape (num_molecules, (num_nbrs+1)*num_beads*num_feature)
    '''
    new_x_shape = np.array(x.shape)
    new_x_shape[1] *= num_nbrs+1
    x_wNbrs = np.zeros(new_x_shape)

    for i in range(len(x)):
        nb_indices = nbrs[i, :num_nbrs+1].astype(int)
        if not i:
            print 'nbrs indices: ', nb_indices
        nb_indices = nb_indices[nb_indices != -1]

        temp_mols = x[nb_indices]
        newshape = (1, np.prod(temp_mols.shape))
        temp_mols = np.reshape(temp_mols, newshape)

        x_wNbrs[i, :temp_mols.shape[1]] = temp_mols

    return x_wNbrs
