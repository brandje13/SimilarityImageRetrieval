import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import dataloader.test_loader as loader
from segment_anything import SamAutomaticMaskGenerator


@torch.no_grad()
def extract_SAM_features(model, data_dir, dataset, gnd_fn, split):
    # 1. Setup Mask Generator
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    all_vectors = []
    all_img_indices = []

    # Construct loader
    test_loader = loader.construct_loader("SAM", data_dir, dataset, gnd_fn, split)

    print(f"Extracting Masked Features for {split}...")

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        # --- ROBUST UNPACKING FIX ---
        # Handle case where loader returns (im, target) tuple vs just Tensor
        if isinstance(batch, torch.Tensor):
            im_batch = batch
        elif isinstance(batch, (list, tuple)):
            im_batch = batch[0]
        else:
            # Fallback: assume the batch itself is the image
            im_batch = batch

        # Ensure it is 4D (Batch, C, H, W)
        if im_batch.dim() == 3:
            im_batch = im_batch.unsqueeze(0)

        # Move to GPU
        im = im_batch.to(device=model.device)

        # 2. Get raw feature grid (1, 256, 64, 64)
        processed_im = model.preprocess(im)
        features = model.image_encoder(processed_im)

        # 3. Generate Masks
        # Needs numpy uint8 input (H, W, 3)
        im_np = im[0].permute(1, 2, 0).cpu().numpy()
        im_np = (im_np * 255).astype(np.uint8)

        masks = mask_generator.generate(im_np)

        if len(masks) == 0:
            # Fallback: Whole image pooling if no masks found
            global_vec = torch.mean(features, dim=(2, 3))
            all_vectors.append(global_vec.cpu())
            all_img_indices.append(batch_idx)
            continue

        # 4. Masked Pooling
        for mask_data in masks:
            binary_mask = torch.from_numpy(mask_data['segmentation']).to(device=model.device)

            # Downsample mask to 64x64
            binary_mask_small = F.interpolate(
                binary_mask.float().unsqueeze(0).unsqueeze(0),
                size=(64, 64), mode='nearest'
            ).squeeze()

            # Weighted Pooling
            mask_expanded = binary_mask_small.unsqueeze(0).expand_as(features[0])
            masked_features = features[0] * mask_expanded

            # Avoid division by zero
            mask_area = mask_expanded.sum(dim=(1, 2)).clamp(min=1.0)
            vector = masked_features.sum(dim=(1, 2)) / mask_area

            all_vectors.append(vector.cpu().unsqueeze(0))
            all_img_indices.append(batch_idx)

    # Cat and Normalize
    if len(all_vectors) > 0:
        all_vectors = torch.cat(all_vectors, dim=0)
        all_vectors = F.normalize(all_vectors, p=2, dim=1)
    else:
        # Failsafe if NO masks found in entire dataset
        print("WARNING: No masks found in any image!")
        all_vectors = torch.zeros((1, 256))

    return all_vectors, torch.tensor(all_img_indices)