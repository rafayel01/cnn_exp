import pickle
import matplotlib.pyplot as plt

with open('/home/rafayel/cnn_exp/results/cifar10_resnet9/ResNet9_par_5_all_list.pkl', 'rb') as f:
    data = pickle.load(f)


# print(data[0])

length: int = len(data[0]['val_acc'])
print(length)

def plot(x: list[float], y: list[float]) -> None:
    plt.plot(x, y)


y = [None for _ in range(6)]
for i in range(6):
    y[i] = [data[i]['val_acc'][j] * 100 for j in range(101)]
    print(max(data[i]['val_acc']))


plot(range(length), y[0])
plot(range(length), y[1])
plot(range(length), y[2])
plot(range(length), y[3])
plot(range(length), y[4])
plot(range(length), y[5])
plt.show()
