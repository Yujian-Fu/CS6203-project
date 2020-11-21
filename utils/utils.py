import math
import os 
import copy
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

import sklearn.metrics.pairwise as smp

import random
from collections import defaultdict

import utils.train as train
import utils.config as config
import utils.csv_record as csv_record
from utils.utils_model import SimpleNet

logger = logging.getLogger("logger")
def train_process(helper):

    similarity_other_path = helper.folder_path + "/model_co_similarity.txt"
    similarity_other_file = open(similarity_other_path, 'w')
    similarity_mean_path = helper.folder_path + "/model_mean_similarity.txt"
    similarity_mean_file = open(similarity_mean_path, 'w')
    write_header = False

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)
    # will that ignore some training process ?
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        # pre-defined or all participant indices
        adversarial_name_keys = []
        # the whole list with all indices 

        ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
        for idx in range(0, len(helper.params['adversary_list'])):
            for ongoing_epoch in ongoing_epochs:
                # poison if the epoch is selected
                # each attacker may have different attack epoch
                if ongoing_epoch in helper.params[str(idx) + '_poison_epochs']:
                    if helper.params['adversary_list'][idx] not in adversarial_name_keys:
                        adversarial_name_keys.append(helper.params['adversary_list'][idx])

        nonattacker=[]
        for adv in helper.params['adversary_list']:
            if adv not in adversarial_name_keys:
                nonattacker.append(copy.deepcopy(adv))
        begnign_num = helper.params['no_models'] - len(adversarial_name_keys)
        random_agent_name_keys = random.sample(helper.begnign_namelist+nonattacker, begnign_num)
        agent_name_keys = adversarial_name_keys + random_agent_name_keys

        #helper.logger.info('----------------------Start of one epoch training----------------------------')
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}. Adversary list: {adversarial_name_keys}')
        # no need for adversarial in the training?
        epochs_submit_update_dict, num_samples_dict = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        logger.info(f'time spent on training: {round(time.time() - t, 2)}')
        #helper.logger.info('----------------------End of one epoch training----------------------------')

        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        
        #layer_analysis(agent_name_keys, adversarial_name_keys, updates, similarity_other_file, similarity_mean_file, write_header, epoch)
        write_header = True

        if helper.params['aggregation_methods'] == 'mean':
            # Average the models
            is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
                                                      target_model=helper.target_model,
                                                      epoch_interval=helper.params['aggr_epoch_interval'])

        #elif helper.params['aggregation_method'] == config.KURM:
            # Use Krum for mitigating the attack

        
        #elif helper.params['aggregation_method'] == config.NORM_CLIPPING:
            # Use norm Clipping for aggregation
        
        elif helper.params['aggregation_methods'] == 'geom_median':
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)

        elif helper.params['aggregation_methods'] == 'foolsgold':
            is_updated, names, weights, alphas = helper.foolsgold_update(helper.target_model, updates)

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = train.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        if helper.params['is_poison']:

            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = train.Mytest_poison(helper=helper,
                                                                                    epoch=temp_global_epoch,
                                                                                    model=helper.target_model,
                                                                                    is_poison=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])


            # test on local triggers
            csv_record.poisontriggertest_result.append(
                ["global", "combine", "", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

            if len(helper.params['adversary_list']) == 1:  # centralized attack
                if helper.params['centralized_test_trigger'] == True:  # centralized attack test on local triggers
                    for j in range(0, helper.params['trigger_num']):
                        trigger_test_byindex(helper, j, epoch)
            else:  # distributed attack
                for agent_name_key in helper.params['adversary_list']:
                    trigger_test_byname(helper, agent_name_key, epoch)

        logger.info(f'Done in {time.time() - start_time} sec.')
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)


    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Model: {helper.params['environment_name']}")
    similarity_other_file.close()
    similarity_mean_file.close()


def layer_analysis(agent_name_keys, adversarial_name_keys, updates, similarity_other_file, similarity_mean_file, write_header, epoch):
    replace_eps = 1e-6
    begnign_name_keys = list(set(agent_name_keys) - set(adversarial_name_keys))
    adversarial_similarity_dict = {}
    begnign_similarity_dict = {}
    begnign_base_dict = {}
    adversarial_mean_dict = {}
    begnign_mean_dict = {}

    #for begnign_key in begnign_name_keys:
    for begnign_key in agent_name_keys:
        for parameter_name in updates[begnign_key][1]:
            update_array = updates[begnign_key][1][parameter_name].numpy().copy()
            if parameter_name not in begnign_base_dict:
                begnign_base_dict[parameter_name] = update_array
            else:
                begnign_base_dict[parameter_name] = begnign_base_dict[parameter_name] + update_array

            
    for parameter_name in begnign_base_dict:
        begnign_base_dict[parameter_name] = begnign_base_dict[parameter_name] / len(agent_name_keys)
        valid_index = np.where(begnign_base_dict[parameter_name] == 0)
        begnign_base_dict[parameter_name][valid_index] = replace_eps


    # For the adversarial
    for adversarial_key in adversarial_name_keys:
        adversarial_similarity_dict[adversarial_key] = {}
        adversarial_weight = updates[adversarial_key][1]

        for begnign_key in agent_name_keys:
            if adversarial_key != begnign_key:
                begnign_weight = updates[begnign_key][1]

                for parameter_name in adversarial_weight:
                    '''
                    valid_index = np.where(begnign_weight[parameter_name].numpy() != 0)
                    adversarial_weight_array = adversarial_weight[parameter_name].numpy()[valid_index]
                    begnign_weight_array = begnign_weight[parameter_name].numpy()[valid_index]
                    division = (adversarial_weight_array - begnign_weight_array) / begnign_weight_array
                    '''

                    adversarial_weight_array = adversarial_weight[parameter_name].numpy().copy()
                    begnign_weight_array = begnign_weight[parameter_name].numpy().copy()
                    replace_index = np.where(begnign_weight_array == 0)

                    begnign_weight_array[replace_index] = replace_eps
                    adversarial_weight_array[replace_index] = replace_eps
                    division = (adversarial_weight_array - begnign_weight_array) / begnign_weight_array

                    similarity_result = np.mean(np.abs(division))

                    if parameter_name not in adversarial_similarity_dict[adversarial_key]:
                        adversarial_similarity_dict[adversarial_key][parameter_name] = similarity_result
                    else:
                        adversarial_similarity_dict[adversarial_key][parameter_name] += similarity_result

    for adversarial_key in adversarial_name_keys:
        for parameter_name in adversarial_similarity_dict[adversarial_key]:
            adversarial_similarity_dict[adversarial_key][parameter_name] = round(adversarial_similarity_dict[adversarial_key][parameter_name] / (len(agent_name_keys) - 1), 2)

    
    for begnign_key_1 in begnign_name_keys:
        begnign_similarity_dict[begnign_key_1] = {}
        begnign_weight_1 = updates[begnign_key_1][1]

        for begnign_key_2 in agent_name_keys:
            
            if begnign_key_1 != begnign_key_2:
                begnign_weight_2 = updates[begnign_key_2][1]

                for parameter_name in begnign_weight_1:
                    '''
                    valid_index = np.where(begnign_weight_2[parameter_name].numpy != 0)
                    begnign_weight_array_1 = begnign_weight_1[parameter_name].numpy()[valid_index]
                    begnign_weight_array_2 = begnign_weight_2[parameter_name].numpy()[valid_index]
                    division = (begnign_weight_array_1 - begnign_weight_array_2) / begnign_weight_array_2
                    '''
                    begnign_weight_array1 = begnign_weight_1[parameter_name].numpy().copy()
                    begnign_weight_array2 = begnign_weight_2[parameter_name].numpy().copy()
                    replace_index = np.where(begnign_weight_array2 == 0)

                    begnign_weight_array1[replace_index] = replace_eps
                    begnign_weight_array2[replace_index] = replace_eps
                    division = (begnign_weight_array1 - begnign_weight_array2) / begnign_weight_array2

                    similarity_result = np.mean(np.abs(division))

                    if parameter_name not in begnign_similarity_dict[begnign_key_1]:
                        begnign_similarity_dict[begnign_key_1][parameter_name] = similarity_result
                    else:
                        begnign_similarity_dict[begnign_key_1][parameter_name] += similarity_result

    for begnign_key in begnign_name_keys:
        for parameter_name in begnign_similarity_dict[begnign_key]:
            begnign_similarity_dict[begnign_key][parameter_name] = round(begnign_similarity_dict[begnign_key][parameter_name] / (len(agent_name_keys)-1), 2)



    for adversarial_key in adversarial_name_keys:
        adversarial_mean_dict[adversarial_key] = {}
        adversarial_weight = updates[adversarial_key][1]
        for parameter_name in adversarial_weight:
            division = (adversarial_weight[parameter_name].numpy() - begnign_base_dict[parameter_name]) / begnign_base_dict[parameter_name]
            similarity_result = np.mean(np.abs(division))
            if parameter_name not in adversarial_mean_dict[adversarial_key]:
                adversarial_mean_dict[adversarial_key][parameter_name] = similarity_result
            else:
                adversarial_mean_dict[adversarial_key][parameter_name] += similarity_result

    for begnign_key in begnign_name_keys:
        begnign_mean_dict[begnign_key] = {}
        begnign_weight = updates[begnign_key][1]
        for parameter_name in begnign_weight:
            division = (begnign_weight[parameter_name].numpy() - begnign_base_dict[parameter_name]) / begnign_base_dict[parameter_name]
            similarity_result = np.round(np.mean(np.abs(division)), 2)
            if parameter_name not in begnign_mean_dict[begnign_key]:
                begnign_mean_dict[begnign_key][parameter_name] = similarity_result
            else:
                begnign_mean_dict[begnign_key][parameter_name] += similarity_result



    adversarial_layer_similarity = {}
    print("Attacker Record: ")
    for agent_key in adversarial_similarity_dict:
        print(agent_key, adversarial_similarity_dict[agent_key])
        
        for parameter_name in adversarial_similarity_dict[agent_key]:
            
            if parameter_name not in adversarial_layer_similarity:
                adversarial_layer_similarity[parameter_name] = adversarial_similarity_dict[agent_key][parameter_name]
            else:
                adversarial_layer_similarity[parameter_name] += adversarial_similarity_dict[agent_key][parameter_name]
            

    for parameter_name in adversarial_layer_similarity:
        adversarial_layer_similarity[parameter_name] = round(adversarial_layer_similarity[parameter_name] / (len(adversarial_name_keys)), 2)


    print("Average attacker record")
    print(adversarial_layer_similarity)


    begnign_layer_similarity = {}
    print("Bengin Record: ")
    for agent_key in begnign_similarity_dict:
        print(agent_key, begnign_similarity_dict[agent_key])
        for parameter_name in begnign_similarity_dict[agent_key]:
            if parameter_name in begnign_layer_similarity:
                begnign_layer_similarity[parameter_name] += begnign_similarity_dict[agent_key][parameter_name]
            else:
                begnign_layer_similarity[parameter_name] = begnign_similarity_dict[agent_key][parameter_name]

    for parameter_name in begnign_layer_similarity:
        begnign_layer_similarity[parameter_name] = round(begnign_layer_similarity[parameter_name] / (len(begnign_name_keys)), 2)


    if not write_header:
        similarity_other_file.write("Epoch  Agent  ")
        similarity_mean_file.write("Epoch  Agent  ")

        for parameter_name in begnign_layer_similarity:
            similarity_other_file.write(parameter_name + "  ")
            similarity_mean_file.write(parameter_name + "  ")
        similarity_other_file.write("\n")
        similarity_mean_file.write("\n")


    similarity_other_file.write("Attacker: \n")
    for agent_key in adversarial_similarity_dict:
        similarity_other_file.write(str(epoch) + "  " + str(agent_key) + "  ")
        for parameter_name in adversarial_similarity_dict[agent_key]:
            similarity_other_file.write(str(adversarial_similarity_dict[agent_key][parameter_name]) + " ")
        similarity_other_file.write("\n")
    
    similarity_other_file.write("Attacker Average: \n      ")
    for parameter_name in adversarial_layer_similarity:
        similarity_other_file.write(str(adversarial_layer_similarity[parameter_name]) + "  ")
    similarity_other_file.write("\n")

    similarity_other_file.write("Begnign Worker: \n")
    for agent_key in begnign_similarity_dict:
        similarity_other_file.write(str(epoch) + "  " + str(agent_key) + "  ")
        for parameter_name in begnign_similarity_dict[agent_key]:
            similarity_other_file.write(str(begnign_similarity_dict[agent_key][parameter_name]) + "  ")
        similarity_other_file.write("\n")
    
    similarity_other_file.write("Begnign Average: \n      ")
    for parameter_name in begnign_layer_similarity:
        similarity_other_file.write(str(begnign_layer_similarity[parameter_name]) + "  ")
    similarity_other_file.write("\n")

    similarity_mean_file.write("Attacker: \n")
    for agent_key in adversarial_mean_dict:
        similarity_mean_file.write(str(epoch) + "  " + str(agent_key) + " ")
        for parameter_name in adversarial_mean_dict[agent_key]:
            similarity_mean_file.write(str(adversarial_mean_dict[agent_key][parameter_name]) + "  ")
        similarity_mean_file.write("\n")

    similarity_mean_file.write("Begnign Worker: \n")
    for agent_key in begnign_mean_dict:
        similarity_mean_file.write(str(epoch) + "  " + str(agent_key) + " ")
        for parameter_name in begnign_mean_dict[agent_key]:
            similarity_mean_file.write(str(begnign_mean_dict[agent_key][parameter_name]) + "  ")
        similarity_mean_file.write("\n")


    print("Average begnign record")
    print(begnign_layer_similarity)

    print("Attack mean record: ")
    for agent_key in adversarial_mean_dict:
        print(agent_key, adversarial_mean_dict[agent_key])
    print("Begnign mean record: ")
    for agent_key in begnign_mean_dict:
        print(agent_key, begnign_mean_dict[agent_key])

''''''


def trigger_test_byindex(helper, index, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        train.Mytest_poison_trigger(helper=helper, model=helper.target_model,
                                   adver_trigger_index=index)
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_index_" + str(index) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])


def trigger_test_byname(helper, agent_name_key, epoch):
    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
        train.Mytest_poison_agent_trigger(helper=helper, model=helper.target_model, agent_name_key=agent_name_key)
    
    csv_record.poisontriggertest_result.append(
        ['global', "global_in_" + str(agent_name_key) + "_trigger", "", epoch,
         epoch_loss, epoch_acc, epoch_corret, epoch_total])


class Helper:
    def __init__(self, current_time, name, parameters):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None
        self.best_loss = math.inf

        self.params = parameters
        self.name = name
        self.train_dataset = None
        self.test_dataset = None
        self.folder_path = f'saved_models/{self.name}/{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold()

    # set all values to zero to the model parameter
    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator


    def load_data(self):

        # {classes_indices (0, 10): image indices (0, 50000)}
        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')

        ## sample indices for participants using Dirichlet distribution
        # Indices_per_participant []
        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['number_of_total_participants'], #100
            alpha=self.params['dirichlet_alpha'])
        train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                            indices_per_participant.items()]

        # train_loaders [id (1, parts), indices_list]
        logger.info('train loaders done')
        self.train_data = train_loaders

        # All the test data
        self.test_data = self.get_test()

        # (data to be poisoned, data not to be poisoned) 
        self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()

        # predefined attacker id list?
        self.advasarial_namelist = self.params['adversary_list']

        # randomly choose the parts id
        # why call random ? Not random

        # the real participant will be selected
        self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)
        # divided into 2 parts: good list or bad list
        self.begnign_namelist =list(set(self.participants_list) - set(self.advasarial_namelist))


    def model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        if self.params['aggregation_methods'] == 'foolsgold':
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates

        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])
                        update[name].add_(local_model_update_dict[name])
                        detached_data= data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data=detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name]=detached_data # from gpu to cpu

                updates[state_keys[i]]=(num_samples,update)

            return weight_accumulator,updates

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            if update_per_layer.dtype!=data.dtype:
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        return True

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def get_test(self):
        # get the whole 
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader


    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000 = 50000 / 10
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        # []
        image_nums = []

        for n in range(no_classes):
            image_num = []
            random.shuffle(cifar_classes[n])
            # alpha is the proportion of total size used for training
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                # the number of images with label n for user to train
                no_imgs = int(round(sampled_probabilities[user]))
                # the indices list
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                # image_num: [num_parts, [the list for label n for each participant]]
                image_num.append(len(sampled_list))
                # add the indices
                per_participant_list[user].extend(sampled_list)
                # users are not using the same picture for twice
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)

        # {use_id, user_indices_list}
        return per_participant_list

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        # get all the image data for a given user
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True, num_workers=8)
        return train_loader


    def poison_test_dataset(self):
            logger.info('get poison test loader')
            # delete the test data with chosen target label
            test_classes = {}

            # classify the test data with label
            # {(id, indices)}
            for ind, x in enumerate(self.test_dataset):
                _, label = x
                if label in test_classes:
                    test_classes[label].append(ind)
                else:
                    test_classes[label] = [ind]

            range_no_id = list(range(0, len(self.test_dataset)))
            for image_ind in test_classes[self.params['poison_label_swap']]:
                if image_ind in range_no_id:
                    # no indices in the original list any more
                    range_no_id.remove(image_ind)
            
            poison_label_inds = test_classes[self.params['poison_label_swap']]
            # divide the original indice dataset into 2 parts: poison or not

            return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=self.params['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                range_no_id)), \
                torch.utils.data.DataLoader(self.test_dataset,
                                                batch_size=self.params['batch_size'],
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    poison_label_inds))

    def dp_noise(self, param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False):
        images, targets = bptt
        poison_count= 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.params['poisoning_per_batch']:
                    new_targets[index] = self.params['poison_label_swap']
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(config.device)
        new_targets = new_targets.to(config.device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image,adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns= []
        if adversarial_index==-1:
            for i in range(0,self.params['trigger_num']):
                poison_patterns = poison_patterns+ self.params[str(i) + '_poison_pattern']
        else :
            poison_patterns = self.params[str(adversarial_index) + '_poison_pattern']
        if self.params['type'] == 'cifar' or self.params['type'] == 'tiny-imagenet':
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1


        elif self.params['type'] == 'mnist':

            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1

        else:
            print("Type error")
            exit(0)

        return image

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(config.device)
        target = target.to(config.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target


class FoolsGold(object):
    def __init__(self):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads


        wv, alpha = self.foolsgold(self.memory)  # Use FG

        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    
    def foolsgold(self,grads):
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha














