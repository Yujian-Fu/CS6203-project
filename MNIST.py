import datetime
from MNIST_utils import MNIST
from paras.Mnist import base
from paras.Mnist import cen_sing, cen_multi
from paras.Mnist import multi_cen_sing, multi_cen_multi
from paras.Mnist import diff_privacy_multi_001, diff_privacy_multi_002
from paras.Mnist import dis_multi, dis_sing
from paras.Mnist import foolsgold, geomedian
from paras.Mnist import double_pixel_multi, double_pixel_sing
from paras.Mnist import half_attack_multi, half_attack_sing
from utils.csv_record import clear_all_record

import utils.utils
import logging
import argparse

mode_list = ["cen", "multi_cen", "dis", "double_pix", "defense", "half_attack", "similarity", "all"]

logger = logging.getLogger("main logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def run_test(parameters, current_time, folder_name, similarity_test = False):
    clear_all_record()
    helper = MNIST(current_time, folder_name, parameters)
    helper.create_model(similarity_test)
    helper.load_data()
    utils.utils.train_process(helper, similarity_test)

def test_conf(base_para, choice_para, name, similarity_test = False):
    base_para.update(choice_para)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    run_test(base_para, current_time, "mnist_" + name, similarity_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mode')
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()
    
    mode = args.params
    check_correctness = False

    if mode == "cen" or mode == "all":
        check_correctness = True
        logger.info("Testing Centralized Single-Shot on Mnist")
        test_conf(base.parameter_base, cen_sing.parameters, "cen_sing")

        logger.info("Testing Centralized Multi-Shot on Mnist")
        test_conf(base.parameter_base, cen_multi.parameters, "cen_multi")


    if mode == "multi_cen" or mode == "all":
        check_correctness = True
        logger.info("Testing Multi-Distributed Single-Shot on Mnist")
        test_conf(base.parameter_base, multi_cen_sing.parameters, "multi_cen_sing")

        logger.info("Testing Multi-Distributed Multi-Shot on Mnist")
        test_conf(base.parameter_base, multi_cen_multi.parameters, "multi_cen_multi")


    if mode == "dis" or mode == "all":
        check_correctness = True
        logger.info("Testing Distributed Single-Shot on Mnist")
        test_conf(base.parameter_base, dis_sing.parameters, "dis_sing")

        logger.info("Testing Distributed Multi-Shot on Mnist")
        test_conf(base.parameter_base, dis_multi.parameters, "dis_multi")

    '''
    if mode == "diff" or mode == "all":
        check_correctness = True
        logger.info("Testing Diff-Privacy Single-Shot on Mnist")
        test_conf(base.parameter_base, diff_privacy_multi_001.parameters, "diff-001")

        logger.info("Testing Distributed Multi-Shot on Mnist")
        test_conf(base.parameter_base, diff_privacy_multi_002.parameters, "diff-002")
    '''
    
    if mode == "double_pix" or mode == "all":
        check_correctness = True
        logger.info("Testing Double Pixel Single-Shot on Mnist")
        test_conf(base.parameter_base, double_pixel_sing.parameters, "double_pix_sing")

        logger.info("Testing Double Pixel Multi-Shot on Mnist")
        test_conf(base.parameter_base, double_pixel_multi.parameters, "double_pix_multi")
    
    if mode == "defense" or mode == "all":
        check_correctness = True
        logger.info("Testing Double Pixel Multi-Shot on Mnist")
        test_conf(base.parameter_base, foolsgold.parameters, "foolsgold")

        logger.info("Testing GeoMedian Single-Shot on Mnist")
        test_conf(base.parameter_base, geomedian.parameters, "geomedian")

    if mode == "half_attack" or mode == "all":
        check_correctness = True
        logger.info("Testing Half Attack Single-Shot on Mnist")
        test_conf(base.parameter_base, half_attack_sing.parameters, "half_attack_sing")

        logger.info("Testing Half Attack Multi-Shot on Mnist")
        test_conf(base.parameter_base, half_attack_multi.parameters, "half_attack_multi")

    if mode == "similarity":
        check_correctness = True
        logger.info("Testing similarity on dis multi and cen multi shot setting")
        test_conf(base.parameter_base, cen_multi.parameters, "cen_multi_simi", True)

        #test_conf(base.parameter_base, dis_multi.parameters, "dis_multi_simi", True)

        


    if not check_correctness:
        print("Mode Error! Choose from these setting: ", mode_list)
    

        



    















