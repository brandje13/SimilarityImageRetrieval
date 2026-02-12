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

    text = '>> {}: Image Retrieval with Segment Anything'.format(dataset)
    print(text)

    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, "SAM_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_SAM_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, "SAM_data_features.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_SAM_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    # 2. Flatten Spatial Grid: (N, 256, 4096)
    # We turn every image into a "Bag of 4096 Patches"
    N_q, C, H, W = Q.shape
    N_db = X.shape[0]

    Q_flat = torch.from_numpy(Q).to(device).view(N_q, C, -1)  # (7, 256, 4096)
    X_flat = torch.from_numpy(X).to(device).view(N_db, C, -1)  # (1046, 256, 4096)

    # 3. Normalize Patterns (Cosine Similarity prep)
    # We normalize every patch individually so magnitude doesn't matter, only pattern.
    Q_flat = F.normalize(Q_flat, dim=1)
    X_flat = F.normalize(X_flat, dim=1)

    print("Running Patch-to-Patch Search (Max-Max)...")

    # We need to store scores for all pairs
    all_scores = np.zeros((N_q, N_db), dtype=np.float32)

    # 4. The Search Loop
    # We loop through queries to save memory.
    # Calculating (4096 x 4096) matrix for every image pair is heavy!

    for i in tqdm(range(N_q), desc="Queries"):
        # Get one Query's patches: (256, 4096)
        q_patches = Q_flat[i]

        # Compare against ALL Database images
        # To go faster, we can batch the database, but let's do simple loop for safety
        for j in range(N_db):
            x_patches = X_flat[j]  # (256, 4096)

            # SIMILARITY MATRIX: (4096 Query Patches) x (4096 Db Patches)
            # Result: How similar is every patch in Q to every patch in X[j]?
            # Shape: (4096, 4096)
            sim_matrix = torch.mm(q_patches.t(), x_patches)

            # THE MAGIC: "Best Patch Match"
            # 1. For every patch in Q, find its best match in X[j] (Max over cols)
            # 2. Then take the average (or max) of those best matches.

            # Option A: Absolute Max (Risky - prone to single-pixel noise)
            # score = sim_matrix.max()

            # Option B: "Chamfer-style" (Robust)
            # "How well is the Query covered by this image?"
            # We find the best match for every query patch, then average the top 10% best matches.
            # This ignores the "boring wall patches" that match poorly, and focuses on the "good stuff".

            best_match_per_patch, _ = sim_matrix.max(dim=1)  # (4096,)

            # Sort scores and take top k (e.g., top 100 patches matching)
            # This focuses the score on the "Object" and ignores the "Background"
            k = 100
            top_k_scores, _ = torch.topk(best_match_per_patch, k)
            score = top_k_scores.mean()

            all_scores[i, j] = score.item()

    # 5. Rank Results
    # Argsort gives low->high, so we reverse it for similarity
    # We want indices of the highest scores
    ranks = np.argsort(-all_scores, axis=1).T

    if True:
        # revisited evaluation
        ks = [10, 25, 100]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

            print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                         np.around(mapM * 100, decimals=2),
                                                                         np.around(mapH * 100, decimals=2)))

    return ranks
