import pickle
import matplotlib.pyplot as plt
n_epochs=100
models = ("ResNet18_with_cable_eq_bn", "ResNet18_with_cable_eq_2_bn", "ResNet18_with_cable_eq_3_bn", "ResNet18_with_cable_eq_4_bn", "ResNet18_with_cable_eq_5_bn")
model_cable_eq_dict = []
with open(f'/home/rafayel/cnn_exp/results/model_original.pkl', 'rb') as fp:
  model_cable_eq_dict.append(pickle.load(fp))
for model in models:
    with open(f'/home/rafayel/cnn_exp/results/with_bn/{model}.pkl', 'rb') as fp:
      model_cable_eq_dict.append(pickle.load(fp))
      #print(model_cable_eq_dict)
      #print('dictionary opened successfully to file')

for m in range(len(model_cable_eq_dict)):
  print(f"Accuracy with {m} cable eq layer: {max(model_cable_eq_dict[m]['test_acc'])}%")

lst = ('train_loss', 'train_acc', 'test_loss', 'test_acc')
for mod in lst:
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[0][mod], label="Original")
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[1][mod], label="With 1 Cable eq layer")
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[2][mod], label="With 2 Cable eq layer")
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[3][mod], label="With 3 Cable eq layer")
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[4][mod], label="With 4 Cable eq layer")
  plt.plot(range(1, n_epochs + 1), model_cable_eq_dict[5][mod], label="With 5 Cable eq layer")
  plt.title("Training Loss vs Num of Epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.show()