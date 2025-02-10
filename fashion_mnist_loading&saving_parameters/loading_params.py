# here we will load the saved parameters from the existing file

import numpy as np
import cv2
import os
from network_full import *

# dataset loading
def load_data_mnist(dataset,path):
    labels=os.listdir(os.path.join(path,dataset))
    X=[]
    Y=[]

    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image=cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)

            X.append(image)
            Y.append(label)
    return np.array(X) , np.array(Y).astype('uint8')

def create_data_mnist(path):
    
    x_train,y_train=load_data_mnist('test',path)
    return x_train,y_train

x_test,y_test= create_data_mnist('fashion_mnist_images')

#data normalisation
x_test=(x_test.astype(np.float32)-127.5)/127.5

#data flattening

x_test=x_test.reshape(x_test.shape[0],-1)



#model building and training
model=Model()
model.add(layer_dense(x_test.shape[1],128))
model.add(Activation_ReLu())
model.add(layer_dense(128,128))
model.add(Activation_ReLu())
model.add(layer_dense(128,10))
model.add(Activation_SoftMax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          accuracy=Accuracy_Categorical(),
          )


model.finalize()

#loading the parameters so we don't have to train again 
model.load_params('fashion_mnist.params')
#testing
model.evaluate(x_test,y_test,batch_size=128)

