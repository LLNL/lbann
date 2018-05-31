# copied originally from p2b1_mol_AE.py:Candle_Molecular_Train
# modifications:
#  * removed train_ac(), format_data()
#  * removed class members not required for datagen
#  * removed internal randomization
#  * removed debug print()s
#  * removed all customization except molecular_nbrs
import helper
import numpy as np


class Candle_Molecular_Datagen():
    def __init__(self):
        self.numpylist, _ = helper.get_local_files('3k_run16')
        self.molecular_nbrs = 10

    def datagen(self):
        X_all = np.array([])
        nbrs_all = np.array([])
        resnums_all = np.array([])
        files = self.numpylist
        # Training only on few files
        order = [0]
        # Randomize files after first training epoch

        for f_ind in order:
            (X, nbrs, resnums) = helper.get_data_arrays(files[f_ind])

            # normalizing the location coordinates and bond lengths and scale type encoding
            # Changed the xyz normalization from 255 to 350
            Xnorm = np.concatenate([X[:, :, :, 0:3] / 320., X[:, :, :, 3:8], X[:, :, :, 8:] / 10.], axis=3)

            num_frames = X.shape[0]
            input_feature_dim = np.prod(Xnorm.shape[2:])

            xt_all = np.array([])
            yt_all = np.array([])

            for i in range(num_frames):

                xt = Xnorm[i]
                xt = helper.get_neighborhood_features(xt, nbrs[i], self.molecular_nbrs)

                yt = xt.copy()
                #xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)

                if not len(xt_all):
                    xt_all = np.expand_dims(xt, axis=0)
                    yt_all = np.expand_dims(yt, axis=0)
                else:
                    xt_all = np.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
                    yt_all = np.append(yt_all, np.expand_dims(yt, axis=0), axis=0)

            yield files[f_ind], xt_all, yt_all
        return
