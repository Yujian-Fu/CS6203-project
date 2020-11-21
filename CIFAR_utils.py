import sys
sys.path.append("..")
from utils.utils import Helper
from utils.utils_model import ResNet, BasicBlock
from utils.config import device
import torch
import random
import logging
from torchvision import datasets, transforms


logger = logging.getLogger("logger")

class CIFAR(Helper):

    def create_model(self):
        self.local_model = ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(self.name), created_time=self.created_time)
        
        self.target_model = ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(self.name), created_time=self.created_time)

       # Caution! this is used in CPU !
        self.local_model= self.local_model.to()
        self.target_model= self.target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            self.target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        logging.info("Loading CIFAR Images")

        self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),]))
        self.test_dataset = datasets.CIFAR10('./data', train=False, 
                            transform=transforms.Compose([transforms.ToTensor(),]))











