import pickle
import matplotlib.pyplot as plt

with open('/home/rafayel/cnn_exp/results/cifar10_resnet9/ResNet9_only_exp_par_1.pkl', 'rb') as f:
    data = pickle.load(f)


#print(data)

length: int = len(data['val_acc'])
# print(length)

def plot(x: list[float], y: list[float]) -> None:
    plt.plot(x, y)


# y = [None for _ in range(6)]
# for i in range(6):
#     y[i] = [data[i]['val_acc'][j] * 100 for j in range(101)]
#     print(data[i]['val_acc'][-1])

print(data['val_acc'])
print(max(data['val_acc']))
plot(range(length), data['val_acc'])
plt.show()
