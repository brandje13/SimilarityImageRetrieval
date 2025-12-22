# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import torch
import numpy as np
from tkfilebrowser import askopenfilenames, askopendirname

from test.config_gnd import config_gnd
from test.test_utils import extract_feature, test_revisitop, print_top_n, create_groundtruth, \
    create_groundtruth_from_txt
from test.dataset import DataSet

from modules.reranking.MDescAug import MDescAug
from modules.reranking.RerankwMDA import RerankwMDA


@torch.no_grad()
def test_model(model, device, data_dir, dataset_list, scale_list, custom, update_data, update_queries,
               is_rerank, gemp, rgem, sgem, onemeval, depth, evaluate, logger):
    torch.backends.cudnn.benchmark = False
    model.eval()
    torch.cuda.set_device(device)
    state_dict = model.state_dict()

    # initialize modules
    MDescAug_obj = MDescAug(M=600, K=9)
    RerankwMDA_obj = RerankwMDA(M=600, K=9)

    model.load_state_dict(state_dict)

    for dataset in [dataset_list]:
        text = '>> {}: Global Retrieval for scale {} with CVNet-Global'.format(dataset, str(scale_list))
        print(text)
        if custom:
            query_paths = askopenfilenames()
            data_dir = askopendirname()
            create_groundtruth(query_paths, data_dir, dataset)  # TODO: Fix custom dataset param
            gnd_fn = 'custom.json'
            dataset = "custom"
        elif dataset in ['roxford5k', 'rparis6k']:
            gnd_fn = f'gnd_{dataset}.json'
            file_path = os.path.join(data_dir, dataset)
            create_groundtruth_from_txt(data_dir, dataset)
        elif not dataset == "":
            query_paths = [os.path.join(data_dir, dataset, "queries", i)
                           for i in os.listdir(os.path.join(data_dir, dataset, "queries"))]
            create_groundtruth(query_paths, data_dir, dataset)
            file_path = os.path.join(data_dir, dataset)
            gnd_fn = f'gnd_{dataset}.json'
        else:
            file_path = ''
            assert dataset

        cfg = config_gnd(dataset, data_dir, custom)

        print("extract query features")
        Q_path = os.path.join(data_dir, dataset, "query_features.pt")
        if update_queries or not os.path.isfile(Q_path):
            Q = extract_feature(model, data_dir, dataset, gnd_fn, "query", [1.0], gemp, rgem, sgem, scale_list)
            torch.save(Q, Q_path)
        else:
            Q = torch.load(Q_path)

        print("extract database features")
        X_path = os.path.join(data_dir, dataset, "data_features.pt")
        if update_data or not os.path.isfile(X_path):
            X = extract_feature(model, data_dir, dataset, gnd_fn, "db", [1.0], gemp, rgem, sgem, scale_list)
            torch.save(X, X_path)
        else:
            X = torch.load(X_path)

        # Q = torch.tensor(Q).cuda()
        # X = torch.tensor(X).cuda()

        Q_tensor = torch.from_numpy(Q).float().to(device)
        X_tensor = torch.from_numpy(X).float().to(device)

        print("perform global feature reranking")
        if onemeval:
            X_expand = torch.load(f"./feats_1m_RN{depth}.pth").cuda()
            X = torch.cat([X, X_expand], 0)
        # sim = torch.matmul(X_tensor, Q_tensor.T)
        top_k = 10
        # # ranks = torch.argsort(-sim, dim=0)
        # vals, ranks = torch.topk(sim, k=top_k, dim=0)

        # Build Index (Flat = Brute Force, IP = Inner Product)
        # 2048 is the dimension of your features
        index = faiss.IndexFlatIP(2048)

        # Add Gallery (X) to Index
        index.add(X_tensor.cpu().numpy())

        # Search
        # D = distances (values), I = indices (ranks)
        D, I = index.search(Q_tensor.cpu().numpy(), top_k)

        # 'I' is your 'ranks' variable
        ranks = I.T  # Transpose to match your original shape (Gallery, Query)

        if is_rerank:
            ranks = torch.from_numpy(ranks).to(device)
            rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba = MDescAug_obj(X_tensor, Q_tensor, ranks)
            ranks = RerankwMDA_obj(ranks, rerank_dba_final, res_top1000_dba, ranks_trans_1000_pre, x_dba)
        ranks = ranks.data.cpu().numpy()

        if evaluate:
            # TODO: Move to main call
            print_top_n(cfg, ranks, 10, file_path)

            # revisited evaluation
            ks = [10, 25, 100]
            if not custom:
                (mapE, _, _, _), (mapM, _, _, _), (mapH, _, _, _) = test_revisitop(cfg, ks, [ranks, ranks, ranks])

                print('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                          np.around(mapM * 100, decimals=2),
                                                                          np.around(mapH * 100, decimals=2)))
                logger.info('Retrieval results {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE * 100, decimals=2),
                                                                                np.around(mapM * 100, decimals=2),
                                                                                np.around(mapH * 100, decimals=2)))

    return ranks

