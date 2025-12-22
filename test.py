# written by Seongwon Lee (won4113@yonsei.ac.kr)

import config as config
import core.CVNet_tester as CVNet_tester

from config import cfg


def main():
    config.load_cfg_fom_args("test a CVNet model.")
    cfg.NUM_GPUS = 1
    cfg.freeze()

    # TODO: Move groundtruth here and pass to main
    gnd = []

    SG_ranks = CVNet_tester.__main__(gnd)

    # TODO: retrieve top imgs set
    # TODO: print_top_k

    # TODO: SAM

    # TODO: retrieve top imgs set
    # TODO: print_top_k

    # TODO: Union or Intersection

    # TODO: Show final set


if __name__ == "__main__":
    main()
