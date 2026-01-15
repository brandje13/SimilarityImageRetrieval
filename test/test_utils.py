# written by Seongwon Lee (won4113@yonsei.ac.kr)
import os
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import test.test_loader as loader
from test.evaluate import compute_map
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from PIL import Image
import json


@torch.no_grad()
def extract_feature(model, data_dir, dataset, gnd_fn, split, scale_list_fix, gemp, rgem, sgem, scale_list):
    with torch.no_grad():
        test_loader = loader.construct_loader(model.model, data_dir, dataset, gnd_fn, split, scale_list_fix)
        img_feats = [[] for _ in range(len(scale_list_fix))]

        for im_list in tqdm(test_loader):
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()

                desc = model.extract_global_descriptor(im_list[idx], gemp, rgem, sgem, scale_list)

                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                img_feats[idx].append(desc.detach().cpu())

        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats = img_feats[0]  # 6422 2048

        img_feats = F.normalize(img_feats, p=2, dim=1)
        img_feats = img_feats.cpu().numpy()

    return img_feats


@torch.no_grad()
def test_revisitop(cfg, ks, ranks):
    # revisited evaluation
    gnd = cfg['gnd']
    ranks_E, ranks_M, ranks_H = ranks

    # evaluate ranks
    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['good'], gnd[i]['ok']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks_E, gnd_t, cfg, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['ok']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['ok']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks_M, gnd_t, cfg, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['ok']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['good']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks_H, gnd_t, cfg, ks)

    return (mapE, apsE, mprE, prsE), (mapM, apsM, mprM, prsM), (mapH, apsH, mprH, prsH)


def mix_score(sim_ori, sim_corr, ratio):
    return sim_ori + sim_corr * ratio


@torch.no_grad()
def rerank_ranks_revisitop(cfg, topk, ranks, sim_global, sim_local_dict, mix_ratio=0.5):
    gnd = cfg['gnd']
    ranks_corr_E = ranks.copy()
    ranks_corr_M = ranks.copy()
    ranks_corr_H = ranks.copy()
    for i in range(int(cfg['nq'])):
        rerank_count_E = 0
        rerank_count_M = 0
        rerank_count_H = 0
        rerank_idx_list_E = []
        rerank_idx_list_M = []
        rerank_idx_list_H = []
        rerank_rank_idx_list_E = []
        rerank_rank_idx_list_M = []
        rerank_rank_idx_list_H = []
        rerank_score_list_E = []
        rerank_score_list_M = []
        rerank_score_list_H = []
        append_E = False
        append_M = False
        append_H = False
        for j in range(int(cfg['n'])):
            rank_j = ranks[j][i]
            if rank_j in gnd[i]['junk']:
                append_E = False
                append_M = False
                append_H = False
                continue
            elif rank_j in gnd[i]['good']:
                append_E = True
                append_M = True
                append_H = False
            elif rank_j in gnd[i]['ok']:
                append_E = False
                append_M = True
                append_H = True
            else:  # negative
                append_E = True
                append_M = True
                append_H = True

            if rerank_count_E >= topk:
                append_E = False
            if rerank_count_M >= topk:
                append_M = False
            if rerank_count_H >= topk:
                append_H = False

            if not append_E and not append_M and not append_H:
                continue
            if append_E:
                rerank_count_E += 1
                rerank_idx_list_E.append(j)
                rerank_rank_idx_list_E.append(rank_j)
                rerank_score_list_E.append(sim_local_dict[(rank_j, i)])
            if append_M:
                rerank_count_M += 1
                rerank_idx_list_M.append(j)
                rerank_rank_idx_list_M.append(rank_j)
                rerank_score_list_M.append(sim_local_dict[(rank_j, i)])
            if append_H:
                rerank_idx_list_H.append(j)
                rerank_count_H += 1
                rerank_rank_idx_list_H.append(rank_j)
                rerank_score_list_H.append(sim_local_dict[(rank_j, i)])

        rerank_score_np_E = np.asarray(rerank_score_list_E)
        sim_query_E = sim_global[rerank_rank_idx_list_E, i]
        rerank_score_np_mix_E = mix_score(sim_query_E, rerank_score_np_E, mix_ratio)
        topk_ranks_corr_E = np.argsort(-rerank_score_np_mix_E)
        ranks_corr_E[rerank_idx_list_E, i] = np.asarray(rerank_rank_idx_list_E)[topk_ranks_corr_E]

        rerank_score_np_M = np.asarray(rerank_score_list_M)
        sim_query_M = sim_global[rerank_rank_idx_list_M, i]
        rerank_score_np_mix_M = mix_score(sim_query_M, rerank_score_np_M, mix_ratio)
        topk_ranks_corr_M = np.argsort(-rerank_score_np_mix_M)
        ranks_corr_M[rerank_idx_list_M, i] = np.asarray(rerank_rank_idx_list_M)[topk_ranks_corr_M]

        rerank_score_np_H = np.asarray(rerank_score_list_H)
        sim_query_H = sim_global[rerank_rank_idx_list_H, i]
        rerank_score_np_mix_H = mix_score(sim_query_H, rerank_score_np_H, mix_ratio)
        topk_ranks_corr_H = np.argsort(-rerank_score_np_mix_H)
        ranks_corr_H[rerank_idx_list_H, i] = np.asarray(rerank_rank_idx_list_H)[topk_ranks_corr_H]

    return ranks_corr_E, ranks_corr_M, ranks_corr_H


# TODO: Move to separate file
def retrieve_and_print_top_n(cfg, ranks, n, retrieve_only=True):
    ranks = np.transpose(ranks)
    images = []
    top_n = {}
    file_path = cfg['path']
    resize_transform = transforms.Resize((224, 224))

    for i in range(len(ranks)):
        query = cfg['qimlist'][i]
        top_n[query] = {'query': str, 'top_n': []}
        top_n[query]['query'] = query
        if not retrieve_only:
            image = read_image(os.path.join(file_path, "queries", query))
            image = resize_transform(image)
            images.append(image)

        for j in range(n):
            next_best = cfg['imlist'][ranks[i][j]]
            top_n[query]['top_n'].append(os.path.join(file_path, next_best))
            if not retrieve_only:
                image = read_image(os.path.join(file_path, next_best))
                image = resize_transform(image)
                images.append(image)

    if not retrieve_only:
        images_tensor = torch.stack(images)
        grid = make_grid(images_tensor, nrow=n + 1)
        img = torchvision.transforms.ToPILImage()(grid)
        img.show()

    return top_n


# TODO: Move to separate file
def create_groundtruth(query_paths, dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Determine the category based on the filename
    # parts = filename.split('_')
    if os.name == 'nt':
        split = "\\"
    elif os.name == 'posix':
        split = "/"
    else:
        split = "_"  # TODO: Better fix for this

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg, .jpeg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg"):
            # Add the file to the list
            data['imlist'].append(img)

    if all(os.path.isdir(path) for path in query_paths) and dataset == "ILIAS":
        for path in query_paths:
            temp_queries = []
            temp_path = os.path.join(path, "query")

            # Iterate through each file in the query directory
            for file in sorted(os.listdir(temp_path)):
                # Check if the file ends with .jpg, .jpeg or .png
                # Leave extentions to handle multiple data types
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    query_file = os.path.join(path.split('\\')[-1], 'query', file)
                    query_name = file.split('.')[0]
                    temp_queries.append(query_file)

                    if query_name not in query_info:
                        bbx_path = os.path.join(path, 'query', query_name + '_bbox.txt')
                        with open(bbx_path, 'r') as f:
                            content = f.read().strip()
                            raw_bbx = list(map(float, content.split()))
                            x, y, w, h = raw_bbx
                            x2 = x + w
                            y2 = y + h

                            bbx = [x, y, x2, y2]

                        query_info[query_file] = {'query': query_file, 'bbx': bbx,
                                                  'ok': [], 'good': [], 'junk': []}
                        data['qimlist'].append(query_file)

            # Iterate through each file in the pos directory
            temp_path = os.path.join(path, "pos")
            for file in sorted(os.listdir(temp_path)):
                # Check if the file ends with .jpg, .jpeg or .png
                # Leave extentions to handle multiple data types
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    pos_file = os.path.join("queries", path.split('\\')[-1], 'pos', file)
                    data['imlist'].append(pos_file)

                    for img in temp_queries:
                        query_info[img]['ok'].append(pos_file)
    else:
        # Iterate over all image files in the directory
        for filename in sorted(query_paths):
            query_name = filename.split(split)[-1]  # Extract query name
            category = 'query'

            # Process each line in the text file
            if query_name not in query_info:
                query_info[query_name] = {'query': None, 'bbx': None,
                                          'ok': [], 'good': [], 'junk': []}

            # Check if the line indicates a query
            if category == 'query':
                query_info[query_name][category] = query_name
                w, h = Image.open(filename).size
                query_info[query_name]['bbx'] = [0, 0, w, h]
                data['qimlist'].append(query_name)

            # Populate data dictionary based on category
            if category in ['ok', 'good', 'junk']:
                query_info[query_name][category].append(None)

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


# Create groundtruth based on provided txt files
def create_groundtruth_from_txt(dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg"):
            # Add the file to the list
            data['imlist'].append(img)

    # Iterate over all text files in the directory
    for filename in sorted(os.listdir(os.path.join(dir_path, dataset, "groundtruth"))):
        if filename.endswith('.txt'):
            # Determine the category based on the filename
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])  # Extract query name
            category = parts[-1][:-4]

            # Read the content of the text file
            with open(os.path.join(os.path.join(dir_path, dataset, "groundtruth"), filename), 'r') as file:
                lines = file.readlines()

            # Process each line in the text file
            for line in lines:
                parts = line.split()

                if query_name not in query_info:
                    query_info[query_name] = {'query': None, 'bbx': None,
                                              'ok': [], 'good': [], 'junk': []}

                # Check if the line indicates a query
                if category == 'query':
                    query_info[query_name][category] = parts[0][5:] + '.jpg'
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(parts[0][5:] + '.jpg')
                    shutil.copy(os.path.join(dir_path, dataset, parts[0][5:] + '.jpg'), os.path.join(dir_path, dataset, "queries"))

                # Populate data dictionary based on category
                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data
