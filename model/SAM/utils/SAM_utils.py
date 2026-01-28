import torch
from tqdm import tqdm

import dataloader.test_loader as loader

@torch.no_grad()
def extract_SAM_features(model, data_dir, dataset, gnd_fn, split):
    im_feats = []
    test_loader = loader.construct_loader("SAM", data_dir, dataset, gnd_fn, split)
    for im_batch in tqdm(test_loader):
        im = im_batch.to(device=model.device)
        im = model.preprocess(im)
        feats = model.image_encoder(im)
        desc = torch.mean(feats, dim=(2, 3))
        im_feats.append(desc.detach().cpu())

    return im_feats
