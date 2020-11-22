# CS6203-project
This is the project for CS6203 in NUS
Environment Requirements:
Pytorch

How to run the experiments:
Get the result on CIFAR and MNIST dataset:
CIFAR run seperately : 
python CIFAR.py --params cen
python CIFAR.py --params multi_cen 
python CIFAR.py --params dis 
python CIFAR.py --params double_pix 
python CIFAR.py --params half_attack
python CIFAR.py --params defense

Run all above:
python CIFAR.py --params all

change CIFAR.py to MNIST.py to get the result on MNIST dataset

Visualizatioin:
mkdir saved_figures
mkdir saved_figures/mnist
mkdir saved_figures/cifar

Visualize the CIFAR results:
python CIFAR_visualization.py --params all

Visualize the MNIST results:
python MNIST_visualization.py --params all

You will find the figures in "saved_figures/cifar" and "saved_figures/mnist" respectively

Repos for reference:
https://github.com/AI-secure/DBA
https://github.com/ebagdasa/backdoor_federated_learning
https://github.com/krishnap25/RFA
https://github.com/DistributedML/FoolsGold

