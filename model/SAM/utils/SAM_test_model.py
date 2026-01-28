import os
import torch
from model.SAM.utils.SAM_utils import extract_SAM_features


@torch.no_grad()
def test_SAM(model, device, cfg, gnd, data_dir, dataset, custom, update_data, update_queries):
    torch.backends.cudnn.benchmark = True
    model.eval()
    torch.cuda.set_device(device)

    text = '>> {}: Image Retrieval with Segment Anything'.format(dataset)
    print(text)

    print("extract query features")
    Q_path = os.path.join(data_dir, dataset, "SAM_query_features.pt")
    if update_queries or not os.path.isfile(Q_path):
        Q = extract_SAM_features(model, data_dir, dataset, gnd, "query")
        torch.save(Q, Q_path)
    else:
        Q = torch.load(Q_path)

    print("extract database features")
    X_path = os.path.join(data_dir, dataset, "SAM_data_features.pt")
    if update_data or not os.path.isfile(X_path):
        X = extract_SAM_features(model, data_dir, dataset, gnd, "db")
        torch.save(X, X_path)
    else:
        X = torch.load(X_path)

    Q_tensor = torch.from_numpy(Q).float().to(device)
    X_tensor = torch.from_numpy(X).float().to(device)

    ranks = 0
    return ranks


