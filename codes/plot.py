import pickle


with open('/home/rafayel/cnn_exp/results/cifar10_resnet9/ResNet9_list.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)