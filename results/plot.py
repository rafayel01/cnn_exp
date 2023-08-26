import pickle
import matplotlib.pyplot as plt

models = ("ResNet18_bn", "ResNet18_with_cable_eq_bn", "ResNet18_with_cable_eq_2_bn", "ResNet18_with_cable_eq_3_bn", "ResNet18_with_cable_eq_4_bn", "ResNet18_with_cable_eq_5_bn")
data =[]
for model in models:
    with open(f'/home/rafayel/cnn_exp/results/cifar10/with_relu_bn/lr_0.5/{model}.pkl', 'rb') as f:
        data.append(pickle.load(f))

length: int = len(data[-1]['test_acc'])
for i in range(len(data)):
    print(max(data[i]["test_acc"]))
    plt.plot(range(length), data[i]["test_acc"])


plt.show()

#print(data)

# length: int = len(data['val_acc'])
# # print(length)

# def plot(x: list[float], y: list[float]) -> None:
#     plt.plot(x, y)


# # y = [None for _ in range(6)]
# # for i in range(6):
# #     y[i] = [data[i]['val_acc'][j] * 100 for j in range(101)]
# #     print(data[i]['val_acc'][-1])

# print(data['val_acc'])
# print(max(data['val_acc']))
# plot(range(length), data['val_acc'])
# plt.show()
