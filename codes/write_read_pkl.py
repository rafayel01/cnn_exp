import pickle 

#d = [{'val': [1, 2, 3], 'test': [4, 5, 6]}, {'val': [1, 20, 30], 'test': [4, 5, 10]}]
with open('filename.pkl', 'rb') as file:
    d = pickle.load(file)

print(d)