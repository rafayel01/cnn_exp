import pickle
import matplotlib.pyplot as plt

with open('/home/rafayel/cnn_exp/results/cifar10_resnet9/ResNet9_par_5_list.pkl', 'rb') as f:
    data = pickle.load(f)


# print(data[0])

length: int = len(data[0]['val_acc'])

def plot(x: list[float], y: list[float]) -> None:
    plt.plot(x, y)
    plt.show()


for i in range(6):
    y = [data[i]['val_acc'][j] * 100 for j in range(100)]
    plot(range(length), data[i]['val_acc'])