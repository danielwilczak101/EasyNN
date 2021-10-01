from EasyNN.dataset.mnist import  dataset, model,show

# Downloads dataset to computer
train_data,train_labels,test_data,test_labels = dataset

print(model(test_data[6]))

show(test_data[6])
