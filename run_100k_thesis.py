import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil
import pickle
import gc
import torch

import config as config
from config import cfg as c

import model.SuperGlobal.CVNet_tester as CVNet_tester
from model.DINOv2 import DINO_tester
from model.SigLIP import SigLIP_tester

from utils.config_gnd import config_gnd
from utils.evaluate_final import evaluate_final
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_top_k, save_merged_results
from utils.merge_results import merge_results


def load_ckpt(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_ckpt(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def archive_output_folder(dataset_path, run_tag):
    """Moves generated grids and raw_sets.txt from output/ into output_<run_tag>/ to prevent overwrites."""
    src_out = os.path.join(dataset_path, "output")
    dst_out = os.path.join(dataset_path, f"output_{run_tag}")
    if os.path.exists(src_out):
        if os.path.exists(dst_out):
            shutil.rmtree(dst_out)
        shutil.copytree(src_out, dst_out)
        shutil.rmtree(src_out)
        print(f">> Archived visual grids & sets to: {dst_out}")


def evaluate_and_save_combo(cfg, models_list, modes, run_tag):
    """Fuses a specific ensemble across requested modes and archives its output folder. Skips if already done."""
    dst_out = os.path.join(cfg['path'], f"output_{run_tag}")
    if os.path.exists(dst_out):
        print(f">> Skipping {run_tag} (Already completed at {dst_out})")
        return

    print(f"\n{'=' * 60}\nEVALUATING CONFIG: {run_tag}\n{'=' * 60}")
    for mode in modes:
        print(f"\n--- Mode: {mode.upper()} ({run_tag}) ---")
        merged = merge_results(cfg, models_list, mode)
        save_merged_results(cfg, merged, models_list, mode)
        evaluate_final(cfg, models_list, merged, mode)
    archive_output_folder(cfg['path'], run_tag)


def main():
    config.load_cfg_fom_args("Execute 100k Thesis Final Evaluations")
    c.NUM_GPUS = 1

    # --- 1. SETUP GROUND TRUTH ---
    if c.TEST.DATASET in ['roxford5k', 'rparis6k']:
        gnd = f'gnd_{c.TEST.DATASET}.json'
        create_groundtruth_from_txt(c.TEST.DATA_DIR, c.TEST.DATASET)
    elif c.TEST.DATASET != "":
        gnd = f'gnd_{c.TEST.DATASET}.json'
        gnd_path = os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, gnd)
        if not os.path.exists(gnd_path):
            query_paths = [os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries", i)
                           for i in sorted(os.listdir(os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "queries")))]
            create_groundtruth(query_paths, c.TEST.DATA_DIR, c.TEST.DATASET)
    else:
        raise ValueError("Please specify TEST.DATASET (e.g., TEST.DATASET \"'ILIAS_100k'\")")

    cfg = config_gnd(c.TEST.DATASET, c.TEST.DATA_DIR, False, gnd)

    ckpt_dir = os.path.join(c.TEST.DATA_DIR, c.TEST.DATASET, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ranks_ckpt_path = os.path.join(ckpt_dir, "thesis_final_ranks.pkl")
    cached_ranks = load_ckpt(ranks_ckpt_path)

    # --- 2. EXTRACT / RERANK CORE MODELS ONCE ---
    # A. SigLIP SO400m 384 (Used in ALL configurations)
    siglip_key = "siglip_so400m_384"
    if siglip_key not in cached_ranks:
        print("\n--- Executing SigLIP (google/siglip-so400m-patch14-384) ---")
        c.SigLIP.WEIGHTS = "google/siglip-so400m-patch14-384"
        c.SigLIP.RESOLUTION = 384
        ranks_siglip, map_siglip = SigLIP_tester.__main__(gnd, cfg)
        cached_ranks[siglip_key] = ranks_siglip
        save_ckpt(cached_ranks, ranks_ckpt_path)
    else:
        print(">> Loaded SigLIP SO400m 384 ranks from checkpoint.")
        ranks_siglip = cached_ranks[siglip_key]

    # B. DINOv2 Giant reg4 336 @ M=1000 (2-Way Pareto Winner & Controlled 3-Way)
    dino336_m1000_key = "dino_giant_reg4_336_M1000"
    if dino336_m1000_key not in cached_ranks:
        print("\n--- Executing DINOv2 Giant reg4 336 (M=1000) ---")
        c.DINO.WEIGHTS = "vit_giant_patch14_reg4_dinov2.lvd142m"
        c.DINO.RESOLUTION = 336
        c.DINO.TOP_M = 1000
        ranks_dino336_m1000, _ = DINO_tester.__main__(gnd, cfg)
        cached_ranks[dino336_m1000_key] = ranks_dino336_m1000
        save_ckpt(cached_ranks, ranks_ckpt_path)
    else:
        print(">> Loaded DINOv2 Giant 336 (M=1000) ranks from checkpoint.")
        ranks_dino336_m1000 = cached_ranks[dino336_m1000_key]

    # C. DINOv2 Giant reg4 336 @ M=2000 (Reuses cached 336 .h5 file on disk! Zero re-extraction!)
    dino336_m2000_key = "dino_giant_reg4_336_M2000"
    if dino336_m2000_key not in cached_ranks:
        print("\n--- Executing DINOv2 Giant reg4 336 (M=2000, Warm H5 Cache) ---")
        c.DINO.WEIGHTS = "vit_giant_patch14_reg4_dinov2.lvd142m"
        c.DINO.RESOLUTION = 336
        c.DINO.TOP_M = 2000
        c.TEST.UPDATE_DATA = False
        c.TEST.UPDATE_QUERIES = False
        ranks_dino336_m2000, _ = DINO_tester.__main__(gnd, cfg)
        cached_ranks[dino336_m2000_key] = ranks_dino336_m2000
        save_ckpt(cached_ranks, ranks_ckpt_path)
    else:
        print(">> Loaded DINOv2 Giant 336 (M=2000) ranks from checkpoint.")
        ranks_dino336_m2000 = cached_ranks[dino336_m2000_key]

    # D. SuperGlobal CVNet_R101 @ M=300 (For 3-Way Baselines)
    sg_m300_key = "cvnet_r101_M300"
    if sg_m300_key not in cached_ranks:
        print("\n--- Executing SuperGlobal CVNet_R101 (M=300) ---")
        c.TEST.WEIGHTS = os.path.join("weights", "CVPR2022_CVNet_R101.pyth")
        c.MODEL.DEPTH = 101
        c.SupG.TOP_M = 300
        ranks_sg_m300, _ = CVNet_tester.__main__(gnd, cfg)
        cached_ranks[sg_m300_key] = ranks_sg_m300
        save_ckpt(cached_ranks, ranks_ckpt_path)
    else:
        print(">> Loaded SuperGlobal CVNet_R101 (M=300) ranks from checkpoint.")
        ranks_sg_m300 = cached_ranks[sg_m300_key]

    # --- 3. GENERATE VISUAL GRIDS & METRICS FOR CORE CONFIGS ---
    # Only draw standalone Top-10 model grids if output_single_models_K10 doesn't exist yet
    single_out = os.path.join(cfg['path'], "output_single_models_K10")
    retrieve_only_singles = os.path.exists(single_out)

    topk_siglip_10 = retrieve_top_k(cfg, ranks_siglip, 10, "SigLIP_SO400m_384", retrieve_only_singles)
    topk_dino336_m1000_10 = retrieve_top_k(cfg, ranks_dino336_m1000, 10, "DINO_Giant_336_M1000", retrieve_only_singles)
    topk_dino336_m2000_10 = retrieve_top_k(cfg, ranks_dino336_m2000, 10, "DINO_Giant_336_M2000", retrieve_only_singles)
    topk_sg_m300_10 = retrieve_top_k(cfg, ranks_sg_m300, 10, "CVNet_R101_M300", retrieve_only_singles)
    if not retrieve_only_singles:
        archive_output_folder(cfg['path'], "single_models_K10")

    # Config 1: Global Pareto Winner (2-Way Union & Intersection, 336px, M=1000, K=10)
    models_2way_336_k10 = [
        ["DINOv2_Giant_336_M1000", topk_dino336_m1000_10],
        ["SigLIP_SO400m_384", topk_siglip_10]
    ]
    evaluate_and_save_combo(cfg, models_2way_336_k10, ['union', 'intersection'], "2way_Pareto_336_M1000_K10")

    # Config 2: Controlled 3-Way Ablation (CVNet M=300 + DINO 336 M=1000 + SigLIP, K=10)
    models_3way_m1000_k10 = [
        ["SuperGlobal_R101_M300", topk_sg_m300_10],
        ["DINOv2_Giant_336_M1000", topk_dino336_m1000_10],
        ["SigLIP_SO400m_384", topk_siglip_10]
    ]
    evaluate_and_save_combo(cfg, models_3way_m1000_k10, ['majority', 'intersection', 'union'], "3way_Controlled_336_M1000_K10")

    # Config 3: Best 3-Way Top-K=10 Table Winner (CVNet M=300 + DINO 336 M=2000 + SigLIP, K=10)
    models_3way_m2000_k10 = [
        ["SuperGlobal_R101_M300", topk_sg_m300_10],
        ["DINOv2_Giant_336_M2000", topk_dino336_m2000_10],
        ["SigLIP_SO400m_384", topk_siglip_10]
    ]
    evaluate_and_save_combo(cfg, models_3way_m2000_k10, ['majority', 'intersection', 'union'], "3way_BestK10_336_M2000_K10")

    # Config 4: 3-Way Top-K=50 Majority Benchmark (Zero extra GPU time — sliced from same ranks!)
    if not os.path.exists(os.path.join(cfg['path'], "output_3way_336_M2000_K50")):
        topk_siglip_50 = retrieve_top_k(cfg, ranks_siglip, 50, "SigLIP_SO400m_384", True)
        topk_dino336_m2000_50 = retrieve_top_k(cfg, ranks_dino336_m2000, 50, "DINO_Giant_336_M2000", True)
        topk_sg_m300_50 = retrieve_top_k(cfg, ranks_sg_m300, 50, "CVNet_R101_M300", True)

        models_3way_m2000_k50 = [
            ["SuperGlobal_R101_M300", topk_sg_m300_50],
            ["DINOv2_Giant_336_M2000", topk_dino336_m2000_50],
            ["SigLIP_SO400m_384", topk_siglip_50]
        ]
        evaluate_and_save_combo(cfg, models_3way_m2000_k50, ['majority', 'intersection', 'union'], "3way_336_M2000_K50")
    else:
        print(">> Skipping 3way_336_M2000_K50 (Already completed)")

    # --- 4. OPTIONAL HIGH-RES 518px PARETO KNEE (Runs last so core results are already safe!) ---
    RUN_518_HIGH_RES = True
    if RUN_518_HIGH_RES:
        dino518_m1000_key = "dino_giant_reg4_518_M1000"
        if dino518_m1000_key not in cached_ranks:
            print("\n--- Executing High-Res DINOv2 Giant reg4 518 (M=1000 Pareto Knee) ---")
            c.DINO.WEIGHTS = "vit_giant_patch14_reg4_dinov2.lvd142m"
            c.DINO.RESOLUTION = 518
            c.DINO.TOP_M = 1000
            ranks_dino518_m1000, _ = DINO_tester.__main__(gnd, cfg)
            cached_ranks[dino518_m1000_key] = ranks_dino518_m1000
            save_ckpt(cached_ranks, ranks_ckpt_path)
        else:
            print(">> Loaded DINOv2 Giant 518 (M=1000) ranks from checkpoint.")
            ranks_dino518_m1000 = cached_ranks[dino518_m1000_key]

        single_518_out = os.path.join(cfg['path'], "output_single_DINO518_K10")
        retrieve_only_518 = os.path.exists(single_518_out)
        topk_dino518_m1000_10 = retrieve_top_k(cfg, ranks_dino518_m1000, 10, "DINO_Giant_518_M1000", retrieve_only_518)
        if not retrieve_only_518:
            archive_output_folder(cfg['path'], "single_DINO518_K10")

        models_2way_518_k10 = [
            ["DINOv2_Giant_518_M1000", topk_dino518_m1000_10],
            ["SigLIP_SO400m_384", topk_siglip_10]
        ]
        evaluate_and_save_combo(cfg, models_2way_518_k10, ['union', 'intersection'], "2way_HighRes_518_M1000_K10")


if __name__ == "__main__":
    main()