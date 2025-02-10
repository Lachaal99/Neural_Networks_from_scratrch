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
    X,Y=load_data_mnist('train',path)
    x_train,y_train=load_data_mnist('test',path)
    return X,Y,x_train,y_train

x,y,x_test,y_test= create_data_mnist('fashion_mnist_images')

#data normalisation
x=(x.astype(np.float32)-127.5)/127.5
x_test=(x_test.astype(np.float32)-127.5)/127.5

#data flattening
x=x.reshape(x.shape[0],-1)
x_test=x_test.reshape(x_test.shape[0],-1)

#data shuffling
keys=np.array(range(x.shape[0]))
np.random.shuffle(keys)
x=x[keys]
y=y[keys]

#model building and training
model=Model()
model.add(layer_dense(x.shape[1],128))
model.add(Activation_ReLu())
model.add(Layer_Dropout(0.1))
model.add(layer_dense(128,128))
model.add(Layer_Dropout(0.1))
model.add(Activation_ReLu())
model.add(layer_dense(128,10))
model.add(Activation_SoftMax())

model.set(loss=Loss_CategoricalCrossEntropy(),
          accuracy=Accuracy_Categorical(),
          optimizer=Optimizer_Adam(decay=5e-5)
          )

model.finalize()
#training 
model.train(x,y,validation_data=(x_test,y_test),epochs=5,batch_size=128,print_every=100)

#evaluation
model.evaluate(x_test,y_test,batch_size=128)
#saving the hole model
model.save('fashion_mnist.model')