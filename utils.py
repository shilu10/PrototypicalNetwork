import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def extract_sample(n_way, n_support, n_query, datax, datay):
    """
      Picks random sample of size n_support+n_querry, for n_way classes
      Args:
          n_way (int): number of classes in a classification task
          n_support (int): number of labeled examples per class in the support set
          n_query (int): number of labeled examples per class in the query set
          datax (np.array): dataset of images
          datay (np.array): dataset of labels
      Returns:
          (dict) of:
            (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
            (int): n_way
            (int): n_support
            (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support+n_query)]
        sample.append(sample_cls)
    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.permute(0,1,4,2,3)
    return({
          'images': sample,
          'n_way': n_way,
          'n_support': n_support,
          'n_query': n_query
        })


def euclidean_distance(a, b):
    """
      Computes euclidean distance btw x and y
      Args:
          x (torch.Tensor): shape (n, d). n usually n_way*n_query
          y (torch.Tensor): shape (m, d). m usually n_way
      Returns:
          torch.Tensor: shape(n, m). For each query, the distances to each centroid
    """
    n = a.size(0)
    m = b.size(0)
    d = a.size(1)
    assert d == b.size(1)

    a = a.unsqueeze(1).expand(n, m, d)
    b = b.unsqueeze(0).expand(n, m, d)

    return torch.pow(a - b, 2).sum(2)
