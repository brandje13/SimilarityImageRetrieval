import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import make_grid


def retrieve_and_print_top_k(cfg, ranks, k, retrieve_only=True):
    ranks = np.transpose(ranks)
    images = []
    top_k = {}
    file_path = cfg['path']
    resize_transform = transforms.Resize((224, 224))

    for i in range(len(ranks)):
        query = cfg['qimlist'][i]
        top_k[query] = {'query': str, 'top_k': []}
        top_k[query]['query'] = query
        if not retrieve_only:
            image = read_image(os.path.join(file_path, "queries", query))
            image = resize_transform(image)
            images.append(image)

        for j in range(k):
            next_best = cfg['imlist'][ranks[i][j]]
            top_k[query]['top_k'].append(os.path.join(file_path, next_best))
            if not retrieve_only:
                image = read_image(os.path.join(file_path, next_best))
                image = resize_transform(image)
                images.append(image)

    if not retrieve_only:
        images_tensor = torch.stack(images)
        grid = make_grid(images_tensor, nrow=k + 1)
        img = torchvision.transforms.ToPILImage()(grid)
        img.show()

    return top_k
