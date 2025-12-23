# written by Seongwon Lee (won4113@yonsei.ac.kr)
import os

import config as config
import core.CVNet_tester as CVNet_tester
from tkfilebrowser import askopenfilenames, askopendirname

from config import cfg as c
from test.config_gnd import config_gnd
from test.test_utils import create_groundtruth_from_txt, create_groundtruth, retrieve_and_print_top_n


def main():
    config.load_cfg_fom_args("test a CVNet model.")
    c.NUM_GPUS = 1
    c.freeze()
    cfg = config_gnd(c.TEST.DATASET, c.TEST.DATA_DIR, c.TEST.CUSTOM)

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

    SG_ranks = CVNet_tester.__main__(gnd, cfg)
    SG_top = retrieve_and_print_top_n(cfg, SG_ranks, 10)

    # TODO: SAM

    # TODO: retrieve_and_print_top_k

    # TODO: Union or Intersection

    # TODO: Show final set


if __name__ == "__main__":
    main()
