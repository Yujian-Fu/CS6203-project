parameter_base = {
'type': 'mnist',
'test_batch_size': 64,
'lr': 0.1,
'poison_lr': 0.05,
'poison_step_lr': True,
'momentum': 0.9,
'decay': 0.0005,
'batch_size': 64,
'epochs': 70,
'internal_epochs': 1,
'internal_poison_epochs': 10,
'poisoning_per_batch': 20,
'aggr_epoch_interval': 1, # aggregate every round
'geom_median_maxiter': 10,
'fg_use_memory': True,
'participants_namelist': [0,1,2,3,4,5,6,7,8,9],
'no_models': 10,
'number_of_total_participants': 100,
'is_random_namelist': True,
'is_random_adversary': False,
'is_poison': True,

'environment_name': 'mnist_DBA',

'save_model': False,
'save_on_epochs': [10,16,17,18,19,20,21,22],

'resumed_model': True,
'resumed_model_name': 'mnist_pretrain/model_last.pt.tar.epoch_10',
'resumed_model_path': 'mnist_pretrain/',

'vis_train': False,
'vis_train_batch_loss': False,
'vis_trigger_split_test': True,
'track_distance': False,
'batch_track_distance': False,
'log_interval': 2,
'poison_momentum': 0.9,
'poison_decay': 0.005,
'results_json': False,
'alpha_loss': 1,



'poison_epochs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58 ,59, 60,
                61, 62, 63, 64, 65, 66, 67, 68, 69, 70],


'sampling_dirichlet': True,
'dirichlet_alpha': 0.5,

'scale_weights_poison': 100,
'poison_label_swap': 2,
'centralized_test_trigger': True
}




