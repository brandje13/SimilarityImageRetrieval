# written by Seongwon Lee (won4113@yonsei.ac.kr)
import os

import config as config
import model.SuperGlobal.CVNet_tester as CVNet_tester
from tkfilebrowser import askopenfilenames, askopendirname

from config import cfg as c
from model.SAM import SAM_tester
from utils.config_gnd import config_gnd
from utils.groundtruth import create_groundtruth_from_txt, create_groundtruth
from utils.SIR_topk import retrieve_and_print_top_k


def main():
    config.load_cfg_fom_args("utils a CVNet model.")
    c.NUM_GPUS = 1
    c.freeze()

    if c.TEST.CUSTOM:
        query_paths = askopenfilenames()
        data_dir = askopendirname()
        create_groundtruth(query_paths, data_dir, c.TEST.DATASET)  # TODO: Fix custom dataset param
        gnd = 'custom.json'
        dataset = "custom"
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
    top_k = 10

    #SG_ranks = CVNet_tester.__main__(gnd, cfg)
    #SG_top = retrieve_and_print_top_k(cfg, SG_ranks, top_k, True)

    SAM_ranks = SAM_tester.__main__(gnd, cfg)
    SAM_top = retrieve_and_print_top_k(cfg, SAM_ranks, top_k, True)

    # TODO: Union or Intersection

    # TODO: Show final set


if __name__ == "__main__":
    main()
