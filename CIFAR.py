import datetime
from CIFAR_utils import CIFAR
from paras.cifar import base, cen_sing
import utils.utils

if __name__ == "__main__":

    parameters = base.parameter_base
    

    parameters.update(cen_sing.parameters)

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = CIFAR(current_time, "cifar", parameters)
    helper.create_model()
    helper.load_data()
    utils.utils.train_process(helper)


    















