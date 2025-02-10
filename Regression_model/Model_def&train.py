# Regression problem

import numpy as np
from network import * 
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import sine_data

nnfs.init()

X,y= sine_data()
#model build up
dense1 = layer_dense(1,64)

activation1 = Activation_ReLu()

dense2 =layer_dense(64, 64)

activation2=Activation_ReLu()

dense3= layer_dense(64,1)

activation3=Activation_Linear()

loss_function= Loss_MeanSquaredError()

optimizer= Optimizer_Adam(learning_rate=0.005,decay=1e-3)

accuracy_precision=np.std(y)/250

#training
for epoch in range(10001): 
    #forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    #loss calculation
    data_loss = loss_function.calculate(activation3.output, y)

    regularized_loss= loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)+ loss_function.regularization_loss(dense3)
    
    loss =data_loss + regularized_loss

    predictions =activation3.output
    #metrics calculation
    accuracy= np.mean(np.abs(predictions-y)<accuracy_precision)
    

    if not epoch%100: 
        print(" epoch: %d, acc: %.3f, loss: %.3f,data_loss: %.3f, regularized_loss: %.3f lr: %f"%(epoch, accuracy, loss,data_loss,regularized_loss, optimizer.current_learning_rate) )
    
    #backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # parameters optimization
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# validation & testing
x_test, y_test = sine_data()


dense1.forward(x_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

loss= loss_function.calculate(activation3.output, y_test)

predictions= activation3.output

accuracy = np.mean(np.absolute(predictions-y_test)<accuracy_precision)
print("validation, acc: %.3f , loss: %f"%(accuracy,loss))
   
plt.figure(1)
plt.plot(x_test,y_test)
plt.plot(x_test,predictions)
plt.title("superimposed curves")
plt.figure(2)
plt.plot(x_test,y_test)
plt.title("True Data")
plt.figure(3)
plt.plot(x_test, predictions)
plt.title("Predicted Data")

plt.show()

