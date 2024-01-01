import pickle
import matplotlib.pyplot as plt

models = ("ResNet18_bn", "ResNet18_with_cable_eq_bn", "ResNet18_with_cable_eq_2_bn", "ResNet18_with_cable_eq_3_bn") #, "ResNet18_with_cable_eq_4_bn", "ResNet18_with_cable_eq_5_bn")
models2 = ("ResNet18_with_cable_eq_bn", "ResNet18_with_cable_eq_2_bn", "ResNet18_with_cable_eq_3_bn")

data =[]
data2 = []
for model in models:
    with open(f'/PATH_OF_FILE/{model}.pkl', 'rb') as f:
        data.append(pickle.load(f))


for model in models2:
    with open(f'/home/rafayel/cnn_exp/results/stl10_lr_0.5_blocks_final/{model}.pkl', 'rb') as f:
        data.append(pickle.load(f))

leg = ("Original", "CabEq 1", "CabEq 2", "CabEq 3", "CabEqBl 1", "CabEqBl 2", "CabEqBl 3")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

length: int = len(data[-1]['test_acc'])
all_models = models + models2
for i in range(len(data)):
    print(f'Best Test Acc ({all_models[i]}): {max(data[i]["test_acc"])}')
    ax1.plot(range(length), data[i]["test_acc"], label=leg[i])

ax1.set_xlabel("Number of Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(fontsize=13)

for i in range(len(data)):
    ax2.plot(range(length), data[i]["test_loss"], label=leg[i])

ax2.set_xlabel("Number of Epochs")
ax2.set_ylabel("Loss")


plt.legend(fontsize=13)
plt.show()

