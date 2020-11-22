import datetime
from TINY_utils import TINY
from paras.tiny import base
from paras.tiny import cen_sing, cen_multi
from paras.tiny import multi_cen_sing, multi_cen_multi
from paras.tiny import diff_privacy_multi_001, diff_privacy_multi_002
from paras.tiny import dis_multi, dis_sing
from paras.tiny import foolsgold, geomedian
from paras.tiny import double_pixel_multi, double_pixel_sing
from paras.tiny import half_attack_multi, half_attack_sing
from utils.csv_record import clear_all_record

import utils.utils
import logging
import argparse

mode_list = ["cen", "multi_cen", "diff", "dis", "double_pix", "defense", "half_attack", "all"]

logger = logging.getLogger("main logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def run_test(parameters, current_time, folder_name):
    clear_all_record()
    helper = TINY(current_time, folder_name, parameters)
    helper.create_model()
    helper.load_data()
    utils.utils.train_process(helper)

def test_conf(base_para, choice_para, name):
    base_para.update(choice_para)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    run_test(base_para, current_time, "tiny_" + name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mode')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    
    mode = args.params
    check_correctness = False

    if mode == "cen" or mode == "all":
        check_correctness = True
        logger.info("Testing Centralized Single-Shot on Tiny")
        test_conf(base.parameter_base, cen_sing.parameters, "cen_sing")

        logger.info("Testing Centralized Multi-Shot on Tiny")
        test_conf(base.parameter_base, cen_multi.parameters, "cen_multi")


    if mode == "multi_cen" or mode == "all":
        check_correctness = True
        logger.info("Testing Multi-Distributed Single-Shot on Tiny")
        test_conf(base.parameter_base, multi_cen_sing.parameters, "multi_cen_sing")

        logger.info("Testing Multi-Distributed Multi-Shot on Tiny")
        test_conf(base.parameter_base, multi_cen_multi.parameters, "multi_cen_multi")


    if mode == "dis" or mode == "all":
        check_correctness = True
        logger.info("Testing Distributed Single-Shot on Tiny")
        test_conf(base.parameter_base, dis_sing.parameters, "dis_sing")

        logger.info("Testing Distributed Multi-Shot on Tiny")
        test_conf(base.parameter_base, dis_multi.parameters, "dis_multi")


    if mode == "diff" or mode == "all":
        check_correctness = True
        logger.info("Testing Diff-Privacy Single-Shot on Tiny")
        test_conf(base.parameter_base, diff_privacy_multi_001.parameters, "diff-001")

        logger.info("Testing Distributed Multi-Shot on Tiny")
        test_conf(base.parameter_base, diff_privacy_multi_002.parameters, "diff-002")
    
    if mode == "double_pix" or mode == "all":
        check_correctness = True
        logger.info("Testing Double Pixel Single-Shot on Tiny")
        test_conf(base.parameter_base, double_pixel_sing.parameters, "double_pix_sing")

        logger.info("Testing Double Pixel Multi-Shot on Tiny")
        test_conf(base.parameter_base, double_pixel_multi.parameters, "double_pix_multi")
    
    if mode == "defense" or mode == "all":
        check_correctness = True
        logger.info("Testing GeoMedian Single-Shot on Tiny")
        test_conf(base.parameter_base, geomedian.parameters, "geomedian")

        logger.info("Testing Double Pixel Multi-Shot on Tiny")
        test_conf(base.parameter_base, foolsgold.parameters, "foolsgold")

    if mode == "half_attack" or mode == "all":
        check_correctness = True
        logger.info("Testing Half Attack Single-Shot on Tiny")
        test_conf(base.parameter_base, half_attack_sing.parameters, "half_attack_sing")

        logger.info("Testing Half Attack Multi-Shot on Tiny")
        test_conf(base.parameter_base, half_attack_multi.parameters, "half_attack_multi")

    if not check_correctness:
        print("Mode Error! Choose from these setting: ", mode_list)
    

        



    















