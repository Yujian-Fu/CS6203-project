from utils.utils import Helper
from utils.utils_model import MnistNet
from utils.config import device
import torch
import random
import logging
from torchvision import datasets, transforms


class MNIST(Helper):

    def create_model(self, similarity_test):
        self.local_model = MnistNet(name='local', created_time=self.current_time)
        
        self.target_model = MnistNet(name='target', created_time=self.current_time)

       # Caution! this is used in CPU !
        if similarity_test:
            device = torch.device('cpu')
        self.local_model=self.local_model.to(device)
        self.target_model=self.target_model.to(device)
    
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
            self.target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
        else:
            self.start_epoch = 1

        self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),]))
        self.test_dataset = datasets.MNIST('./data', train=False, 
                               transform=transforms.Compose([
                                   transforms.ToTensor(),]))










