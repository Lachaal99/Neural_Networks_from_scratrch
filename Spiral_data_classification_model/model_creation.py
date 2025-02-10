# Classification problem
from network import *
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
#data loading
X,y= spiral_data(samples=100, classes=3)
x_test,y_test= spiral_data(samples=100,classes=3)
#model build up
model=Model()

model.add(layer_dense(2,512,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4))
model.add(Activation_ReLu())
model.add(Layer_Dropout(0.1))
model.add(layer_dense(512,3))
model.add(Activation_SoftMax())
#model setting
model.set(loss= Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-3),
          accuracy= Accuracy_Categorical()
          )

model.finalize()

#training
model.train(X,y,epochs=10000,print_every=100,validation_data=(x_test,y_test))

#result representation
plt.figure(1)
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap='viridis')
plt.title("validation_data")
plt.figure(2)
plt.scatter(x_test[:,0],x_test[:,1],c=model.predictions,cmap='viridis')
plt.title("predicted_data")
plt.show()