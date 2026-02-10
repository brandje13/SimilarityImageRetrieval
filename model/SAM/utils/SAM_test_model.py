import os

import faiss
import numpy as np
from tqdm import tqdm
import torch
from model.SAM.utils.SAM_utils import extract_SAM_features
import torch.nn.functional as F

from model.SuperGlobal.utils.SG_utils import test_revisitop


@torch.no_grad()
def test_SAM(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, top_k_list):
    torch.backends.cudnn.benchmark = True
    model.eval()
    torch.cuda.set_device(device)

    text = '>> {}: Region-Based Retrieval with Segment Anything'.format(dataset)
    print(text)

    # 1. Extract Query Masks
    print("extract query masks")
    Q_path = os.path.join(data_dir, dataset, "SAM_query_masked.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q_vecs, Q_indices = extract_SAM_features(model, data_dir, dataset, gnd, "query")
        torch.save({'vecs': Q_vecs, 'indices': Q_indices}, Q_path)
    else:
        checkpoint = torch.load(Q_path)
        Q_vecs, Q_indices = checkpoint['vecs'], checkpoint['indices']

    # 2. Extract Database Masks
    print("extract database masks")
    X_path = os.path.join(data_dir, dataset, "SAM_data_masked.pt")
    if update_data or not os.path.isfile(X_path):
        X_vecs, X_indices = extract_SAM_features(model, data_dir, dataset, gnd, "db")
        torch.save({'vecs': X_vecs, 'indices': X_indices}, X_path)
    else:
        checkpoint = torch.load(X_path)
        X_vecs, X_indices = checkpoint['vecs'], checkpoint['indices']

    print(f"Total Query Objects: {Q_vecs.shape[0]}")
    print(f"Total Database Objects: {X_vecs.shape[0]}")

    # Move to GPU
    Q_vecs = Q_vecs.to(device)
    X_vecs = X_vecs.to(device)
    Q_indices = Q_indices.to(device)
    X_indices = X_indices.to(device)

    # 3. Determine Image Counts (for score matrix)
    # We assume indices are 0..N-1. If not, we take max.
    N_q = torch.max(Q_indices).item() + 1
    N_db = torch.max(X_indices).item() + 1

    # Initialize score matrix (Images x Images)
    final_scores = torch.full((N_q, N_db), -1.0, device=device)

    print("Running Mask-to-Mask Search...")

    # 4. The Search Loop (Iterate by Query Image to save memory)
    # Grouping queries allows us to do Q_img_masks x All_DB_Masks

    unique_q_imgs = torch.unique(Q_indices)

    for q_idx in tqdm(unique_q_imgs, desc="Query Images"):
        # Get all masks belonging to this single Query Image
        mask_locs = (Q_indices == q_idx)
        q_vecs_i = Q_vecs[mask_locs]  # (Num_Masks_in_Query, 256)

        # Calculate Sim against ALL Database Objects
        # Shape: (Num_Masks_in_Query, Total_DB_Masks)
        sim_matrix = torch.mm(q_vecs_i, X_vecs.t())

        # Aggregation: "Best Matching Pair"
        # We need to reduce (Num_Q_Masks, Total_DB_Masks) -> (1, N_db)

        # Strategy:
        # 1. For each DB mask, what is the best Q mask? -> Max over rows
        # 2. Then, for each DB Image, what is the best matching DB mask?

        # Optimization: We iterate DB images to aggregate?
        # Or faster: Scatter_reduce? Let's use a robust loop for safety/clarity.

        best_match_per_db_mask, _ = sim_matrix.max(dim=0)  # (Total_DB_Masks,)

        # Now we have the score of every DB object to this Query Image.
        # We must group them by DB Image ID and take the max.

        # Scatter Max: Update final_scores[q_idx, db_img_ids]
        # We use scatter_reduce_ (PyTorch 1.12+) or a loop if older
        try:
            # Modern PyTorch (Fastest)
            final_scores[q_idx].scatter_reduce_(
                0,
                X_indices.long(),
                best_match_per_db_mask,
                reduce="amax",
                include_self=False
            )
        except AttributeError:
            # Fallback for older PyTorch
            for db_vec_idx, score in enumerate(best_match_per_db_mask):
                db_img_id = X_indices[db_vec_idx].item()
                if score > final_scores[q_idx, db_img_id]:
                    final_scores[q_idx, db_img_id] = score

    # 5. Rank Results
    # Move to CPU for numpy operations
    all_scores_np = final_scores.cpu().numpy()

    # Argsort gives low->high, reverse it
    # Transpose to match (N_db, N_q) format
    ranks = np.argsort(-all_scores_np, axis=1).T

    if True:
        # revisited evaluation
        ks = [10, 25, 100]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

            print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                         np.around(mapM * 100, decimals=2),
                                                                         np.around(mapH * 100, decimals=2)))

    return ranks
