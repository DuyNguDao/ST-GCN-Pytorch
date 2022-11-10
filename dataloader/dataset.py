import numpy as np

import torch



def processing_data(features):
    # remove 4 point 0,1,2,3,4 with eye, ear
    # features = np.concatenate([features[:, :, 0:1, :], features[:, :, 5:, :]], axis=2)  # remove point 1,2,3,4
    # ***************************************** NORMALIZE ************************************

    def scale_pose(xy):
        """
        Normalize pose points by scale with max/min value of each pose.
        xy : (frames, parts, xy) or (parts, xy)
        """
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)
        xy_min = np.nanmin(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy_max = np.nanmax(xy, axis=2).reshape(xy.shape[0], xy.shape[1], 1, 2)
        xy = (xy - xy_min) / (xy_max - xy_min) * 2 - 1
        return xy

    features = scale_pose(features)
    # flatten
    # features = features[:, :, :, :].reshape(len(features), features.shape[1], features.shape[2] * features.shape[3])
    return features
