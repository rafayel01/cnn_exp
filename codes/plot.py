import pickle
import matplotlib.pyplot as plt
import numpy as np

models = ("ResNet18_bn", 
          "ResNet18_with_cable_eq_bn", 
          "ResNet18_with_cable_eq_2_bn",
          "ResNet18_with_cable_eq_3_bn",
          "ResNet18_with_cable_eq_4_bn",
          "ResNet18_with_cable_eq_5_bn")
for model in models:
    with open(f'/home/rafayel/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{model}.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"{model[:-3]} best accuracy: {max(data['test_acc'])}%")


plot = ('train_loss', 'train_acc', 'test_loss', 'test_acc')
title = ("Train Loss VS num of Epochs", 
         "Train Accuracy VS num of Epochs",
         "Testing Loss VS num of Epochs",
         "Training Accuracy VS num of Epochs")
ylabel = ("Training Loss", "Training Accuracy", "Testing Loss", "Testing Accuracy")
leg = ("Original",
       "With 1 cable eq layer",
       "With 2 cable eq layer",
       "With 3 cable eq layer",
       "With 4 cable eq layer",
       "With 5 cable eq layer")
colors = ('Black', 'Green', 'Red', 'Yellow', 'Blue', 'Gray')
for i, pl in enumerate(plot):
    for j, model in enumerate(models):
        with open(f'/home/rafayel/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{model}.pkl', 'rb') as f:
            data = pickle.load(f)
        plt.plot(np.arange(1, 101), data[pl], label=leg[j], color=colors[j])
    plt.title(title[i])
    plt.ylabel(ylabel[i])
    plt.xlabel("Num of Epochs")
    plt.legend()
    #plt.savefig(f'/home/rafayel/Paper/CNN/results/stl10/stl10_lr0.5_{pl}.png')
    plt.show()
