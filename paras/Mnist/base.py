---
type: cifar
test_batch_size: 64
lr: 0.1
poison_lr: 0.05

momentum: 0.9
decay: 0.0005
batch_size: 64
epochs: 300
internal_epochs: 2
internal_poison_epochs: 6
internal_poison_clean_epochs: 0
poisoning_per_batch: 5
aggr_epoch_interval: 1
geom_median_maxiter: 10
fg_use_memory: true
participants_namelist: [0,1,2,3,4,5,6,7,8,9] # not used when is_random_namelist == true
no_models: 10
number_of_total_participants: 100
is_random_namelist: true
is_random_adversary: false # fix adversary in their poison epochs
is_poison: true

environment_name: cifar # visdom environment for visualization

save_model: false
save_on_epochs: [250,300,350,450,550,600]

# pretrained clean model:
resumed_model: true
resumed_model_name: cifar_pretrain/model_last.pt.tar.epoch_200

vis_train: false
vis_train_batch_loss: false
vis_trigger_split_test: true
track_distance: false
batch_track_distance: false
log_interval: 2
poison_momentum: 0.9
poison_decay: 0.005
poison_step_lr: true
results_json: true
alpha_loss: 1




poison_epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58 ,59, 60,
                61, 62, 63, 64, 65, 66, 67, 68, 69, 70]

scale_weights_poison: 100
sampling_dirichlet: true
dirichlet_alpha: 0.5
poison_label_swap: 2
centralized_test_trigger: True
