# written by Seongwon Lee (won4113@yonsei.ac.kr)
# refactored for dynamic ensemble parameterization
import os
import sys
from tkfilebrowser import askopenfilenames, askopendirname

import config as config
from config import cfg as c

import model.SuperGlobal.CVNet_tester as CVNet_tester
from model.ConvNeXtV2 import ConvNeXtV2_tester
from model.MixVPR import MixVPR_tester
from model.DINOv2 import DINO_tester
from model.CLIP import CLIP_tester
from model.SigLIP import SigLIP_tester

from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k, save_merged_results
from utils.merge_results import merge_results

TESTER_REGISTRY = {
    'SuperGlobal': CVNet_tester,
    'ConvNeXtV2': ConvNeXtV2_tester,
    'MixVPR': MixVPR_tester,
    'DINOv2': DINO_tester,
    'CLIP': CLIP_tester,
    'SigLIP': SigLIP_tester
}


def main():
    config.load_cfg_fom_args("Execute Dynamic Image Retrieval Ensemble")
    c.NUM_GPUS = 1

    if c.TEST.CUSTOM:
        query_paths = askopenfilenames()
        data_dir = askopendirname()
        create_groundtruth(query_paths, data_dir, c.TEST.DATASET)
        gnd = 'custom.json'
    elif c.TEST.DATASET in ['roxford5k', 'rparis6k']:
        gnd = f'gnd_{c.TEST.DATASET}.json'
        create_groundtruth_from_txt(c.TEST.DATA_DIR, c.TEST.DATASET)
    elif not c.TEST.DATASET == "":
        query_paths = [os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries", i)
                       for i in os.listdir(os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries"))]
        create_groundtruth(query_paths, c.TEST.DATA_DIR, c.TEST.DATASET)
        gnd = f'gnd_{c.TEST.DATASET}.json'
    else:
        assert c.TEST.DATASET

    cfg = config_gnd(c.TEST.DATASET, c.TEST.DATA_DIR, c.TEST.CUSTOM, gnd)

    ensemble_results = []

    for model_cfg in c.ACTIVE_MODELS:
        family = model_cfg["family"]
        if family not in TESTER_REGISTRY:
            print(f"[!] Warning: '{family}' is not in the registry. Skipping.")
            continue

        if family == "SuperGlobal":
            c.TEST.WEIGHTS = model_cfg.get("weight", c.TEST.WEIGHTS)
            c.MODEL.DEPTH = 101 if 'R101' in c.TEST.WEIGHTS else 50
            if "m" in model_cfg: c.SupG.TOP_M = model_cfg["m"]
        elif family == "DINOv2":
            c.DINO.WEIGHTS = model_cfg.get("weight", c.DINO.WEIGHTS)
            c.DINO.RESOLUTION = model_cfg.get("resolution", c.DINO.RESOLUTION)
            if "m" in model_cfg: c.DINO.TOP_M = model_cfg["m"]
        elif family == "SigLIP":
            c.SigLIP.WEIGHTS = model_cfg.get("weight", c.SigLIP.WEIGHTS)
            c.SigLIP.RESOLUTION = model_cfg.get("resolution", c.SigLIP.RESOLUTION)
            if "m" in model_cfg: c.SigLIP.TOP_M = model_cfg["m"]
        elif family == "CLIP":
            c.CLIP.WEIGHTS = model_cfg.get("weight", c.CLIP.WEIGHTS)
            c.CLIP.RESOLUTION = model_cfg.get("resolution", c.CLIP.RESOLUTION)
            if "m" in model_cfg: c.CLIP.TOP_M = model_cfg["m"]
        elif family == "ConvNeXtV2":
            c.ConvNeXtV2.WEIGHTS = model_cfg.get("weight", c.ConvNeXtV2.WEIGHTS)
            c.ConvNeXtV2.RESOLUTION = model_cfg.get("resolution", c.ConvNeXtV2.RESOLUTION)
            if "m" in model_cfg: c.ConvNeXtV2.TOP_M = model_cfg["m"]
        elif family == "MixVPR":
            c.MixVPR.WEIGHTS = model_cfg.get("weight", c.MixVPR.WEIGHTS)
            if "m" in model_cfg: c.MixVPR.TOP_M = model_cfg["m"]

        weight_str = str(model_cfg.get('weight', '')).split('/')[-1].split('\\')[-1].split('.')[0]
        display_name = f"{family}_{weight_str}" if weight_str else family

        if "m" in model_cfg:
            display_name += f"_M{model_cfg['m']}"

        print(f"\n--- Executing {display_name} ---")
        tester = TESTER_REGISTRY[family]

        ranks, map_score = tester.__main__(gnd, cfg)
        top_k_data = retrieve_top_k(cfg, ranks, c.TEST.TOP_K, family, False)

        ensemble_results.append([display_name, top_k_data])

    if not ensemble_results:
        print("[!] No models executed successfully. Exiting.")
        sys.exit(1)

    mode = c.FUSION_MODE
    print(f"\n--- Fusing and Evaluating Mode: {mode.upper()} ---")
    merged = merge_results(cfg, ensemble_results, mode)
    save_merged_results(cfg, merged, ensemble_results, mode)
    evaluate_final(cfg, ensemble_results, merged, mode)


if __name__ == "__main__":
    main()