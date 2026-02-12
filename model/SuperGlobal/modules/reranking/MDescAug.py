import torch
from torch import nn
import numpy as np


class MDescAug(nn.Module):
    """ Top-M Descriptor Augmentation (Batched, Memory Safe, Device Correct) """

    def __init__(self, M=400, K=9, beta=0.15):
        super(MDescAug, self).__init__()
        self.M = M
        self.K = K + 1  # including oneself
        self.beta = beta

    def forward(self, X, Q, ranks):
        # Batch size for processing queries to save memory
        BATCH_SIZE = 50

        N_q = ranks.shape[1]

        rerank_dba_final_all = []
        res_top1000_dba_all = []
        ranks_trans_1000_pre_all = []
        x_dba_all = []

        for i in range(0, N_q, BATCH_SIZE):
            end_i = min(i + BATCH_SIZE, N_q)

            # 1. Slice current batch
            ranks_batch = ranks[:, i:end_i]  # (TopK, Batch)
            Q_batch = Q[i:end_i].cuda()  # (Batch, Dim)

            # 2. Get Top-M Global Indices for this batch
            ranks_trans_batch = torch.transpose(ranks_batch, 1, 0)[:, :self.M]

            # 3. Fetch Neighbors from X
            # Indices on CPU to slice X (if X is on CPU)
            indices = ranks_trans_batch.cpu() if not X.is_cuda else ranks_trans_batch

            # Move features to GPU for math (Approx 250MB per batch)
            X_batch_cuda = X[indices].cuda()

            # 4. Compute Internal Similarities
            # (B, M, D) x (B, M, D) -> (B, M, M)
            res_ie = torch.einsum('abc,adc->abd', X_batch_cuda, X_batch_cuda)

            # 5. Sort Neighbors
            res_ie_ranks = torch.argsort(-res_ie, dim=-1)[:, :, :self.K]
            res_ie_values = torch.gather(res_ie, -1, res_ie_ranks)

            # 6. Apply Beta Weighting
            weights = res_ie_values.clone()
            weights[:, :, 1:] *= self.beta
            weights[:, :, 0] = 1.0
            weights = weights.unsqueeze(-1)  # (B, M, K, 1)

            # 7. Gather Neighbors-of-Neighbors Features
            # Correct offset calculation (Fixed from previous crash)
            B, M, D = X_batch_cuda.shape

            # Flatten to (B*M, D)
            X_flat = X_batch_cuda.view(B * M, D)

            # Create offsets to find neighbors within the flattened batch
            # Offsets: [0, M, 2M, ...] -> Shape (B, 1, 1)
            batch_offsets = (torch.arange(B, device=X_batch_cuda.device) * M).view(B, 1, 1)

            # Calculate final flat indices
            gather_indices = res_ie_ranks + batch_offsets  # (B, M, K)
            gather_indices = gather_indices.view(-1)  # (B*M*K)

            # Gather
            gathered_feats = X_flat[gather_indices]  # (B*M*K, D)
            x_dba_stacked = gathered_feats.view(B, M, self.K, D)

            # 8. Weighted Sum
            numerator = torch.sum(x_dba_stacked * weights, dim=2)
            denominator = torch.sum(weights, dim=2)
            x_dba_augmented = numerator / denominator  # (B, M, D)

            # 9. Query Similarity
            res_top1000_dba = torch.einsum('ac,adc->ad', Q_batch, x_dba_augmented)

            # 10. Local Sort
            ranks_trans_1000_pre = torch.argsort(-res_top1000_dba, dim=-1)

            # 11. Map back to Global IDs
            final_indices_batch = torch.gather(ranks_trans_batch.cuda(), 1, ranks_trans_1000_pre)

            # 12. Store Results
            # FIX: Convert indices to CUDA so RerankwMDA doesn't crash
            for row in final_indices_batch:
                rerank_dba_final_all.append(row.cuda())

            res_top1000_dba_all.append(res_top1000_dba.cuda())
            ranks_trans_1000_pre_all.append(ranks_trans_1000_pre.cuda())
            x_dba_all.append(x_dba_augmented.cuda())

        # Concatenate final results
        # All outputs must be on CUDA for the next module
        res_top1000_dba_ret = torch.cat(res_top1000_dba_all, dim=0)
        ranks_trans_1000_pre_ret = torch.cat(ranks_trans_1000_pre_all, dim=0)
        x_dba_ret = torch.cat(x_dba_all, dim=0)

        return rerank_dba_final_all, res_top1000_dba_ret, ranks_trans_1000_pre_ret, x_dba_ret