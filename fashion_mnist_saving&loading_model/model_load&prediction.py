from network_full import *
import cv2
import os
import numpy as np

fashion_mnist_labels={0:'t_short/top', 1:'trousers',2:'pullover',3:'dress',4:'coat',5:'sandal',6:'shirt',7:'sneaker',8:'bag',9:'ankle boot'}



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

#loading the hole saved model
model= Model.load('fashion_mnist.model')

confidences=model.predict(x_test[:5],batch_size=128)
predictions=model.output_layer_activation.predictions(confidences)

for predict in predictions:
    print(fashion_mnist_labels[predict])

