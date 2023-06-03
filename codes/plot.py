import pickle
import numpy as np
import matplotlib.pyplot as plt

models = ("ResNet18_with_cable_eq_bn", "ResNet18_with_cable_eq_2_bn", "ResNet18_with_cable_eq_3_bn", "ResNet18_with_cable_eq_4_bn", "ResNet18_with_cable_eq_5_bn")

with open('/home/rafayel/cnn_exp/results/stl10/without_bn/ResNet18_bn.pkl', 'rb') as f:
    model_original = pickle.load(f)
    
models_pkl = []
for model in models:
    with open(f'/home/rafayel/cnn_exp/results/stl10/with_relu_bn/{model}.pkl', 'rb') as f:
        models_pkl.append(pickle.load(f))

print(f"Original ResNet18 accuaracy: {max(model_original['test_acc'])}")
for i, mod in enumerate(models_pkl):
    print(f"ResNet18 with {i+1} cabel eq layers accuracy: {max(mod['test_acc'])}")
    
plots = ("train_loss", "train_acc", "test_loss", "test_acc")
ylabel = ("Training Loss", "Training Accuracy", "Testing Loss", "Testing Accuracy")
leg = ("Original", "With 1 cable eq layer", "With 2 cable eq layer", "With 3 cable eq layer", "With 4 cable eq layer", "With 5 cable eq layer")
tit = ("Training Loss VS Num of Epochs", "Training Acc VS Num of Epochs", "Testing Loss VS Num of Epochs", "Testing Acc VS Num of Epochs")
for i, plot in enumerate(plots):
    plt.plot(np.arange(1, 101), model_original[plot], label=leg[0])
    for j, mod in enumerate(models_pkl):
        plt.plot(np.arange(1, 101), mod[plot], label=leg[j+1])
    plt.title(tit[i])
    plt.xlabel("Num of epochs")
    plt.ylabel(ylabel[i])
    plt.legend()
    plt.savefig(f"/home/rafayel/Article_experiments/Paper_CNN/plots/ResNet18_stl10_cable_eq_{plots[i]}")
    plt.show()