import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from model.DINOv2.utils.DINO_utils import extract_DINO_features
from model.SuperGlobal.utils.SG_utils import test_revisitop

@torch.no_grad()
def test_DINO(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries, top_k_list):
    torch.backends.cudnn.benchmark = True
    model.eval()
    # DINO model is already on device from main call usually, but ensure it here
    # model.to(device)

    text = '>> {}: Image Retrieval with DINOv2'.format(dataset)
    print(text)

    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, "DINO_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_DINO_features(model, data_dir, dataset, gnd, "query")
        # Save Protocol 4 for large files
        torch.save(Q, Q_path, pickle_protocol=4)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, "DINO_data_features.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_DINO_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path, pickle_protocol=4)
    else:
        X = torch.load(X_path)

    print(f"Query Shape: {Q.shape}")    # Expect (N_q, 768, 1369)
    print(f"Database Shape: {X.shape}") # Expect (N_db, 768, 1369)

    # 2. Flatten/View
    # Q is already (N, Dim, N_Patches) from our utils
    # We just need to ensure it's on GPU
    N_q, C, N_p = Q.shape
    N_db = X.shape[0]

    Q_flat = torch.from_numpy(Q).to(device)
    X_flat = torch.from_numpy(X).to(device)

    print("Running DINO Patch-to-Patch Search (Max-Max)...")

    # Store scores
    all_scores = np.zeros((N_q, N_db), dtype=np.float32)

    # 4. The Search Loop
    for i in tqdm(range(N_q), desc="Queries"):
        # Get one Query: (Dim, N_Patches)
        q_patches = Q_flat[i]

        for j in range(N_db):
            x_patches = X_flat[j] # (Dim, N_Patches)

            # SIMILARITY MATRIX: (N_Patches_Q) x (N_Patches_DB)
            # q_patches.t() -> (N_p, Dim)
            # x_patches     -> (Dim, N_p)
            # Result        -> (N_p, N_p)
            sim_matrix = torch.mm(q_patches.t(), x_patches)

            # THE MAGIC: "Best Patch Match"
            # For every patch in Query, find best match in Database
            best_match_per_patch, _ = sim_matrix.max(dim=1)

            # Average Top 50% of matches (Robust "Chamfer" Similarity)
            # This ignores the background patches that have low matches
            k = int(N_p * 0.5)
            top_k_scores, _ = torch.topk(best_match_per_patch, k)
            score = top_k_scores.mean()

            all_scores[i, j] = score.item()

    # 5. Rank Results
    # Transpose to (N_Database, N_Queries) for evaluation
    ranks = np.argsort(-all_scores, axis=1).T

    if True:
        ks = [10, 25, 100]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

            print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                         np.around(mapM * 100, decimals=2),
                                                                         np.around(mapH * 100, decimals=2)))

    return ranks