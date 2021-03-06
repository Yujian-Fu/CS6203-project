# CS6203-project
This is the project for CS6203 in NUS

Environment Requirements:

Pytorch, Matplotlib (Tested on Ubuntu18.04 with Pytorch 1.6.0)

**Run the experiments:**

Get the result on CIFAR and MNIST dataset:

**CIFAR run the settings seperately :**

python CIFAR.py --params cen

python CIFAR.py --params multi_cen 

python CIFAR.py --params dis 

python CIFAR.py --params double_pix 

python CIFAR.py --params half_attack

python CIFAR.py --params defense

**Meaning of the parameter: cen for "centralized", multi_cen for "multi_centralized", dis for "distributed", double_pix for "double_pixel", half_attack for "half_attacker", defense for "FoolsGold and Geomedian"**

**Run all above in sequence:**

python CIFAR.py --params all

**change CIFAR.py to MNIST.py to get the result on MNIST dataset, for example:**

python MNIST.py --params all

**Get the result about similarity analysis**

python MNIST.py --params similarity

You will find the record file in saved_models folder -> respective folder with dataset name and parameter setting, in the record folder, "posiontest_result.csv" is the record for target task, "test_result.csv" is the result for main task.

**Visualizatioin:**

mkdir ./saved_figures

mkdir ./saved_figures/mnist

mkdir ./saved_figures/cifar

mkdir ./saved_figures/similarity

**Visualize the CIFAR results:**

python CIFAR_visualization.py --params all

**Visualize the MNIST results:**

python MNIST_visualization.py --params all

**If you only want compare two settings, ensure you have the result of distributed setting (the baseline for all comparison), then add the parameter that you want to compare at behind, for example, to compare centralized setting in MNIST:**

python MNIST_visualization.py --params cen

**Visualize the Similarity Results**

python MNIST_visualization.py --params similarity

**You will find the result figures in "saved_figures/cifar", "saved_figures/mnist", "saved_figures/similarity" respectively**

**Acknowledgement**

**I read, used or adapted some codes from:**

https://github.com/ebagdasa/backdoor_federated_learning

https://github.com/AI-secure/DBA

https://github.com/krishnap25/RFA

https://github.com/DistributedML/FoolsGold

https://github.com/shaoxiongji/federated-learning

https://github.com/FedML-AI/FedML


