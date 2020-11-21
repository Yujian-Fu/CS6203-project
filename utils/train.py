import torch
import copy
import torch.nn as nn
import logging

import utils.csv_record as csv_record

logger = logging.getLogger("train_logger")

def train(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):

    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries=0
    for temp_name in agent_name_keys:
        if temp_name in helper.params['adversary_list']:
            current_number_of_adversaries+=1

    for model_id in range(helper.params['no_models']):
        epochs_local_update_list = []
        last_local_model = dict()
        client_grad = [] # only works for aggr_epoch_interval=1

        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        adversarial_index= -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and agent_name_key in helper.params['adversary_list']:
            for temp_index in range(0, len(helper.params['adversary_list'])):
                if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                    adversarial_index= temp_index
                    localmodel_poison_epochs = helper.params[str(temp_index) + '_poison_epochs']
                    break
            if len(helper.params['adversary_list']) == 1:
                adversarial_index = -1  # the global pattern

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                poison_lr = helper.params['poison_lr']
                # The model may train it for multiple epochs
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']

                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) *internal_epoch_num
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    _, data_iterator = helper.train_data[agent_name_key]
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list=[]
                    for batch_id, batch in enumerate(data_iterator):
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index,evaluation=False)
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)

                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                        loss = helper.params['alpha_loss'] * class_loss + \
                               (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods']== 'foolsgold':
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    if step_lr:
                        scheduler.step()

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size

                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    num_samples_dict[agent_name_key] = dataset_size

                # internal epoch finish

                if not helper.params['baseline']:
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest(helper=helper, epoch=epoch,
                                                                                   model=model, is_poison=False,
                                                                                   agent_name_key=agent_name_key)
                    csv_record.test_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    for key, value in model.state_dict().items():
                        target_value  = last_local_model[key]
                        new_value = target_value + (value - target_value) * clip_rate
                        model.state_dict()[key].copy_(new_value)

            # benign workers
            else:
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    for batch_id, batch in enumerate(data_iterator):

                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch,evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        if helper.params['aggregation_methods'] == 'foolsgold':
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    num_samples_dict[agent_name_key] = dataset_size

                # test local model after internal epoch finishing
                epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest(helper=helper, epoch=epoch,
                                                                               model=model, is_poison=False,
                                                                               agent_name_key=agent_name_key)
                csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            if is_poison:
                if agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = Mytest_poison(helper=helper,
                                                                                          epoch=epoch,
                                                                                          model=model,
                                                                                          is_poison=True,
                                                                                          agent_name_key=agent_name_key)
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                #  test on local triggers
                if agent_name_key in helper.params['adversary_list']:
                    

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                    

                    csv_record.poisontriggertest_result.append(
                        [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                         epoch_acc, epoch_corret, epoch_total])

            # update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            if helper.params['aggregation_methods'] == "foolsgold":
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        epochs_submit_update_dict[agent_name_key] = epochs_local_update_list

    return epochs_submit_update_dict, num_samples_dict


def Mytest(helper, epoch,
           model, is_poison=False, agent_name_key=""):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0

    data_iterator = helper.test_data
    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch,
                  model, is_poison=False, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    data_iterator = helper.test_data_poison
    for batch_id, batch in enumerate(data_iterator):
        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)

        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    data_iterator = helper.test_data_poison
    adv_index = adver_trigger_index
    for batch_id, batch in enumerate(data_iterator):
        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0


    data_iterator = helper.test_data_poison
    adv_index = -1
    for temp_index in range(0, len(helper.params['adversary_list'])):
        if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
            adv_index = temp_index
            break
    for batch_id, batch in enumerate(data_iterator):
        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count

