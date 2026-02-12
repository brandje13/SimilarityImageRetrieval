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

    text = '>> {}: Image Retrieval with DINOv2'.format(dataset)
    print(text)

    # 1. Load Features
    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, "DINO_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_DINO_features(model, data_dir, dataset, gnd, "query")
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

    print(f"Query Shape: {Q.shape}")
    print(f"Database Shape: {X.shape}")

    N_q, C, N_p = Q.shape
    N_db = X.shape[0]

    # --- MEMORY FIX: Keep Database on CPU ---
    # We only move Q to GPU because it's small (1232 images)
    # X stays on CPU (29k images) until needed
    Q_flat = torch.from_numpy(Q).to(device)
    X_flat = torch.from_numpy(X)  # Keep on CPU!

    # Optional: Normalize on CPU to save GPU work
    # (Doing this on CPU is safer for memory)
    X_flat = F.normalize(X_flat, dim=1)
    Q_flat = F.normalize(Q_flat, dim=1)

    print("Running Batched Patch-to-Patch Search...")

    all_scores = np.zeros((N_q, N_db), dtype=np.float32)

    # Search Configuration
    # We process the Database in chunks (batches) to fit in GPU memory
    DB_BATCH_SIZE = 100  # 100 images * 256 patches fits easily in 8GB

    for i in tqdm(range(N_q), desc="Queries"):
        # Get one Query: (1, 256, 768) - Transpose for matmul
        # Q_flat[i] is (768, 256). Transpose -> (256, 768)
        q_patches = Q_flat[i].t().unsqueeze(0)  # (1, 256, 768)

        # Loop through Database in Batches
        for j in range(0, N_db, DB_BATCH_SIZE):
            # 1. Slice Batch
            end = min(j + DB_BATCH_SIZE, N_db)

            # 2. Move Batch to GPU
            # X_batch shape: (B, 768, 256)
            x_batch = X_flat[j:end].to(device)

            # 3. Batched Matrix Multiplication
            # (1, 256, 768) @ (B, 768, 256) -> (B, 256, 256)
            # Result: [Batch, Query_Patch, DB_Patch] similarity
            sim_matrix = torch.matmul(q_patches, x_batch)

            # 4. Max-Max Scoring (Optimized)
            # Find best DB patch match for every Query patch
            # Max over dim 2 (DB_Patches) -> (B, 256)
            best_match_per_patch, _ = sim_matrix.max(dim=2)

            # Average Top 50% matches
            k = int(N_p * 0.5)
            # Topk on dim 1 (Query_Patches) -> (B, k)
            top_k_vals, _ = torch.topk(best_match_per_patch, k, dim=1)

            # Mean score for each image in batch -> (B,)
            batch_scores = top_k_vals.mean(dim=1)

            # 5. Store Results
            all_scores[i, j:end] = batch_scores.cpu().numpy()

    # Rank Results
    ranks = np.argsort(-all_scores, axis=1).T

    if True:
        ks = [10, 25, 100]
        if not custom:
            (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

            print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                         np.around(mapM * 100, decimals=2),
                                                                         np.around(mapH * 100, decimals=2)))

    return ranks
