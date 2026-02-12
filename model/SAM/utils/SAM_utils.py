import torch
from tqdm import tqdm

import torch.nn.functional as F
import dataloader.test_loader as loader

@torch.no_grad()
def extract_SAM_features(model, data_dir, dataset, gnd_fn, split):
    im_feats = []
    test_loader = loader.construct_loader("SAM", data_dir, dataset, gnd_fn, split)
    for im_batch in tqdm(test_loader):
        im = im_batch.to(device=model.device)
        im1 = model.preprocess(im)
        feats = model.image_encoder(im1)
        #desc = torch.mean(feats, dim=(2, 3))
        im_feats.append(feats.detach().cpu())

    im_feats = torch.cat(im_feats, dim=0)
    #im_feats = F.normalize(im_feats, p=2, dim=1)

    return im_feats.numpy()


#def gem_pooling(x, p=3, eps=1e-6):
    # x: (B, C, H, W)
    # p: Power (higher p = closer to Max Pooling, lower p = closer to Average Pooling)
    #return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
