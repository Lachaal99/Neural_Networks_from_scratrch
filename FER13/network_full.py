import numpy as np
import pickle
import copy
# defining the class layer where each layer of our network will be from this class 
class layer_dense: 

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1= 0,bias_regularizer_l1 =0,weight_regularizer_l2 =0,bias_regularizer_l2 =0): 
        self.weights= 0.01* np.random.randn(n_inputs,n_neurons)
        
        self.biases= np.zeros((1,n_neurons))

        self.weight_regularizer_l1=weight_regularizer_l1
        self.bias_regularizer_l1= bias_regularizer_l1
        self.weight_regularizer_l2= weight_regularizer_l2
        self.bias_regularizer_l2= bias_regularizer_l2

    def forward(self, input,training): 
        self.inputs= input
        
        self.output= np.dot(input, self.weights) + self.biases
    
    def backward(self, dvalues): 
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbiases= np.sum(dvalues,axis=0, keepdims=True)

        self.dinputs= np.dot(dvalues, self.weights.T)

        # L1 and L2 regularization expressions in backpropagation
        if self.weight_regularizer_l1>0: 
            dL1= np.ones_like(self.weights)
            dL1[self.weights<0]= -1
            self.dweights += self.weight_regularizer_l1*dL1  

        if self.weight_regularizer_l2>0: 
            self.dweights += 2*self.weight_regularizer_l2*self.weights

        if self.bias_regularizer_l1>0: 
            dL1= np.ones_like(self.biases)
            dL1[self.biases<0]= -1
            self.dbiases += self.bias_regularizer_l1*dL1

        if self.bias_regularizer_l2>0: 
            self.dbiases += 2*self.bias_regularizer_l2*self.biases
    def get_params(self):
        return self.weights,self.biases
    def set_params(self,weights,biases):
        self.weights=weights
        self.biases=biases

# define the dropout layers for our training 
class Layer_Dropout: 

    def __init__(self, rate):
        self.rate = 1-rate

    def forward(self,inputs,training): 
        self.inputs= inputs
        if not training :
            self.output= inputs.copy()
            return 

        self.binary_mask= np.random.binomial(1, self.rate, size=inputs.shape) /self.rate
        self.output = inputs* self.binary_mask

    def backward(self,dvalues): 
        self.dinputs= dvalues*self.binary_mask    

class Layer_Input:

    def forward(self,inputs,training):
        self.output=inputs







#activation functions

# ReLU activation functiono
class Activation_ReLu: 
    def forward(self, inputs,training):

        self.inputs= inputs

        self.output = np.maximum(0,inputs)
        
    
    def backward(self,dvalues):
        self.dinputs= dvalues.copy()


        self.dinputs[self.inputs<=0]=0
    
    def predictions(self,outputs):
        return outputs    

# Softmax activation 

class Activation_SoftMax: 
    
    def forward(self ,inputs,training):
        self.inputs= inputs

        exp_values = np.exp(inputs - np.max(inputs,axis= 1, keepdims=True ))
        probabilities =exp_values/ np.sum(exp_values,axis=1,keepdims=True)
        self.output= probabilities
        
    #backward pass for the softmax activation separetly 
    def backward( self, dvalues): 
        self.dinputs= np.empty_like(dvalues)

        for index , (single_output , single_dvalues) in enumerate(zip(self.output,dvalues)):
            
            single_output = single_output.reshape(-1,1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self,outputs):
       
        return np.argmax(outputs,axis=1)



# Activation sigmoid
class Activation_Sigmoid:
    def forward(self, inputs,training):
        self.inputs= inputs
        self.output= 1/(1+np.exp(-inputs))
   
    def backward(self,dvalues):
        self.dinputs= dvalues*(1-self.output)*self.output  

    def predictions(self,outputs):
        return (outputs>0.5)*1
        
        # activation linear

class Activation_Linear:

    def forward(self,inputs,training):

        self.inputs= inputs
        self.output= inputs

    def backward(self,dvalues):
        self.dinputs= dvalues.copy()
    
    def predictions(self,outputs):
        return outputs


#Optimizers

#SGD optimizer

class optimizer_SGD: 

    def __init__(self, learning_rate=1., decay=0., momentum=0.):

        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay = decay
        self.iterations = 0 
        self.momentum= momentum

    def pre_update_params(self) : 
        if self.decay: 
            self.current_learning_rate= self.learning_rate * (1./ (1. + self.decay *self.iterations))    


    def update_params(self,layer): 
        
        # if we use momentum 
        if self.momentum: 

            # if layer has no momentum arrays create them  
            if not hasattr(layer, 'weight_momentums'): 
                layer.weight_momentums= np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)    


            weight_updates= self.momentum * layer.weight_momentums - self.current_learning_rate*layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases 
            layer.bias_momentums = bias_updates

        else:
            weight_updates= -self.current_learning_rate * layer.dweights
            bias_updates= -self.current_learning_rate* layer.dbiases

        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_update_params(self):
        self.iterations += 1

    # adagrad optimizer
class Optimizer_Adagrad: 

        def __init__(self, learning_rate=1., decay=0.,epsilon= 1e-7):
            self.learning_rate= learning_rate
            self.current_learning_rate= learning_rate
            self.decay = decay 
            self.iterations = 0
            self.epsilon= epsilon

        def pre_update_params(self): 
            self.current_learning_rate= self.learning_rate*(1./(1+ self.decay* self.iterations)) 

        def update_params(self,layer): 

            # if layer does not contain cache arrays, create them filled with  zeros 
            if not hasattr(layer,'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)   
                layer.bias_cache = np.zeros_like(layer.biases)

            #update_cache
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2

            # SGD update + norùmalisation with square rooted cache

            layer.weights += -self.current_learning_rate * layer.dweights /(np.sqrt(layer.weight_cache)+ self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases/ (np.sqrt(layer.bias_cache)+ self.epsilon)


        def post_update_params(self): 
            self.iterations +=1    

class Optimizer_RMSprop: 

    def __init__(self, learning_rate= 0.001, decay = 0., epsilon=1e-7, rho=0.9):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay
        self.iterations =0
        self.epsilon=epsilon
        self.rho= rho

    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate= self.learning_rate*(1./(1. + self.decay*self.iterations))

    def update_params(self, layer): 

            # if layer does not contain cache arrays, create them filled with zeros

        if not hasattr(layer, 'weight_cache'): 
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

            # update the cache 

        layer.weight_cache = self.rho * layer.weight_cache + ( 1- self.rho)* layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho)* layer.dbiases**2             


        # update the weights
        layer.weights+= -self.current_learning_rate* layer.dweights/ (np.sqrt(layer.weight_cache)+self.epsilon)    
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self): 
        self.iterations +=1


# Adaptative momentum optimizer 
class Optimizer_Adam: 
    def __init__(self, learning_rate = 0.001, decay= 0., epsilon= 1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate= learning_rate
        self.current_learning_rate= learning_rate
        self.decay= decay 
        self.iteration = 0 
        self.epsilon = epsilon
        self.beta_1= beta_1
        self.beta_2= beta_2

    def pre_update_params(self): 
        
        self.current_learning_rate= self.learning_rate*(1. / (1. + self.decay * self.iteration))

    def update_params(self, layer):

        if not hasattr(layer,'weight_cache'): 
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        

        layer.weight_momentum= self.beta_1 * layer.weight_momentum + (1-self.beta_1)* layer.dweights
        layer.bias_momentum= self.beta_1*layer.bias_momentum +(1-self.beta_1)* layer.dbiases

        weight_momentum_corrected = layer.weight_momentum /(1 - self.beta_1**(self.iteration+1))
        bias_momentum_corrected = layer.bias_momentum/(1 - self.beta_1**(self.iteration+1))

        layer.weight_cache= self.beta_2* layer.weight_cache + (1- self.beta_2)* (layer.dweights**2)
        layer.bias_cache= self.beta_2 * layer.bias_cache + (1- self.beta_2)* (layer.dbiases**2)

        weight_cache_corrected = layer.weight_cache/(1 - self.beta_2**(self.iteration + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iteration + 1 ))


        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon )
        layer.biases += -self.current_learning_rate* bias_momentum_corrected/ (np.sqrt(bias_cache_corrected)+ self.epsilon)

    def post_update_params(self): 
        self.iteration+=1 


#loss function 
class Loss : 
    
    def regularization_loss( self): 
        regularization_loss= 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0: 
                regularization_loss += layer.weight_regularizer_l1 *np.sum(np.abs(layer.weights))  

            if layer.bias_regularizer_l1 > 0: 
                regularization_loss += layer.bias_regularizer_l1 *np.sum(np.abs(layer.biases))

            if layer.weight_regularizer_l2 > 0: 
                regularization_loss += layer.weight_regularizer_l2 *np.sum(layer.weights* layer.weights)

            if layer.bias_regularizer_l2 > 0: 
                regularization_loss += layer.bias_regularizer_l2 *np.sum(layer.biases* layer.biases)    

        return regularization_loss

    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers= trainable_layers

    def calculate(self,output,y, *,include_regularization=False): 
        sample_loss= self.forward(output,y)
        data_loss= np.mean(sample_loss)

        self.accumulated_sum +=np.sum(sample_loss)
        self.accumulated_count += len(sample_loss)

        if not include_regularization:
            return data_loss
        

        return data_loss ,self.regularization_loss()  
    
    def calculate_accumulated(self,*,include_regularization=False):
        
        data_loss= self.accumulated_sum/self.accumulated_count

        if not include_regularization: 
            return data_loss
        
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum=0
        self.accumulated_count =0

class Loss_CategoricalCrossEntropy(Loss): 



    
    def forward(self, y_pred,y_true): 
        samples= len(y_pred)
        
        y_pred_clipped= np.clip(y_pred,1e-7,1 - 1e-7)

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        
        elif len(y_true.shape)==2: 
            correct_confidences= np.sum(y_pred_clipped*y_true, axis=1)    
        
        negative_log_likelihood = -np.log(correct_confidences)
        
        return negative_log_likelihood
        
    # backward pass for the loss function calculated separetly
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        labels = len(dvalues[0])

        if len( y_true.shape) == 1: 
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues 
        self.dinputs = self.dinputs/samples


#loss function and softmax activation combined

class Activation_Softmax_Loss_CategoricalCrossentropy(): 


        # creates activation and loss function objects
        def __init__(self):
            self.activation = Activation_SoftMax()
            self.loss= Loss_CategoricalCrossEntropy()


        def forward( self, inputs , y_true): 
            
            self.activation.forward(inputs)

            self.output = self.activation.output
            #return losss value 
            return self.loss.calculate(self.output, y_true)   
        
        def backward(self, dvalues, y_true): 
            samples = len(dvalues)

            if len(y_true.shape) == 2: 
                y_true= np.argmax(y_true, axis=1)


            self.dinputs= dvalues.copy()

            self.dinputs[range(samples), y_true]-=1 

            self.dinputs = self.dinputs/samples   

# loss function for the binary cross entropy
class Loss_BinaryCrossentropy(Loss):

    def forward(self,y_pred, y_true):

        y_pred_clipped= np.clip(y_pred, 1e-7,1 -1e-7)

        sample_losses= -(y_true * np.log(y_pred_clipped)) -(1-y_true)*np.log(1-y_pred_clipped)
        sample_losses= np.mean(sample_losses,axis=-1)

        return sample_losses
    
    
    def backward(self,dvalues, y_true):
        
        samples = len(dvalues)

        outputs= len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7,1- 1e-7)
        
        self.dinputs= -( y_true/clipped_dvalues  - (1-y_true)/(1-clipped_dvalues))/outputs
        
        self.dinputs= self.dinputs/samples

# different loss calculation for regression problems 

class Loss_MeanSquaredError(Loss):

    def forward(self,y_pred,y_true):

        sample_Loss= np.mean((y_true-y_pred)**2, axis=-1)

        return sample_Loss

    def backward(self,dvalues, y_true):
        
        samples= len(dvalues)
        
        outputs= len(dvalues[0])
        self.dinputs= -2*(y_true-dvalues)/ outputs
        self.dinputs= self.dinputs/samples

class Loss_MeanAbsoluteError(Loss): 
    
    def forward(self,y_pred, y_true):
        samples_losses= np.mean(np.abs(y_true-y_pred), axis=-1)

        return samples_losses
    
    def backward(self,dvalues,y_true):
        samples= len(dvalues)

        outputs= len(dvalues[0])

        self.dinputs=np.sign((y_true-dvalues))/outputs
        self.dinputs=self.dinputs/samples

#accuracy class
        
class Accuracy:

    def calculate(self,predictions,y):

        comparisons= self.compare(predictions,y)

        accuracy= np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    
    def calculate_accumulated(self):
        
        accuracy= self.accumulated_sum/self.accumulated_count

        return accuracy

    def new_pass(self):
        self.accumulated_sum=0
        self.accumulated_count =0    






#comparison Categorical
# 
class Accuracy_Categorical(Accuracy):

    def init(self,y):
        pass
    def compare(self,predictions,y):
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)

        return predictions == y    
    
# Accuracy regression

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precison=None

    def init(self,y,reinit=False):
        if self.precison is None or reinit:
            self.precison=np.std(y)/250

    def compare(self,predictions,y):
        return np.abs(y-predictions) <self.precision    
    

# Finally Model class definition

class Model: 
    def __init__(self):
        self.layers=[]
        self.softmax_classifier_output= None
        self.history={'accuracy_train':[],'accuracy_val':[],'loss_train':[],'loss_val':[]}

    def add(self,layer):

        self.layers.append(layer)

    def set(self,*,loss=None,optimizer=None,accuracy=None):
        if loss is not None:
            self.loss=loss
        if optimizer is not None:
            self.optimizer=optimizer
        if accuracy is not None:
            self.accuracy= accuracy    

    def finalize(self):
        
        self.layer_input= Layer_Input()

        layer_count= len(self.layers)

        self.trainable_layers= []

        for i in range(layer_count):

            if i ==0:
                self.layers[i].prev = self.layer_input
                self.layers[i].next= self.layers[i+1]

            elif i < layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next= self.layers[i+1]
                
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next= self.loss
                self.output_layer_activation= self.layers[i]

            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])

            self.loss.remember_trainable_layers(self.trainable_layers)        

        if isinstance(self.layers[-1],Activation_SoftMax) and isinstance(self.loss,Loss_CategoricalCrossEntropy): 
            self.softmax_classifier_output= Activation_Softmax_Loss_CategoricalCrossentropy()
        if self.loss is not None:

            self.loss.remember_trainable_layers(self.trainable_layers)




    def evaluate(self,x_val,y_val,*,batch_size=None):
            
            validation_steps =1
            
            if batch_size is not None:

                validation_steps = len(x_val)//batch_size
                
                if validation_steps*batch_size < len(x_val): 
                    validation_steps+=1


            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(validation_steps):

                if batch_size is None:
                    batch_X=x_val
                    batch_y=y_val
                else:
                    batch_X=x_val[step*batch_size:(step+1)*batch_size]
                    batch_y=y_val[step*batch_size:(step+1)*batch_size]

            
                output= self.forward(batch_X,training=False)

                self.loss.calculate(output,batch_y,include_regularization= False)
           
                predictions= self.output_layer_activation.predictions(output) 

                self.accuracy.calculate(predictions,batch_y)
            

            validation_loss= self.loss.calculate_accumulated()
            validation_accuracy=self.accuracy.calculate_accumulated()
            #saving the history of validation metrics
            self.history['accuracy_val'].append(validation_accuracy)
            self.history['loss_val'].append(validation_loss)

            print("validation, acc:%.3f ,loss: %.3f "%(validation_accuracy,validation_loss))



    def train(self,X,y,*,epochs=1,batch_size=None, print_every=1,validation_data=None):
        self.accuracy.init(y)

        train_steps= 1
        
       
        
        if batch_size is not None:
            train_steps=len(X)//batch_size

            if train_steps*batch_size < len(X):
                train_steps+=1

        #main training loop
        for epoch in range(1,epochs+1):

            print("epoch: %d"%(epoch))
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):

                if batch_size is None:
                    batch_X=X
                    batch_y=y
                else:
                    batch_X=X[step*batch_size:(step+1)*batch_size]
                    batch_y=y[step*batch_size:(step+1)*batch_size]


                output= self.forward(batch_X,training=True)

                data_loss , regularization_loss = self.loss.calculate(output,batch_y, include_regularization= True)
                loss = data_loss +regularization_loss

                predictions= self.output_layer_activation.predictions(output) 

                accuracy=self.accuracy.calculate(predictions,batch_y)

                self.backward(output,batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step%print_every or step ==train_steps-1 :
                    print(" step: %d, acc: %.3f, loss: %.3f,data_loss: %.3f, reg_loss: %.3f lr: %f"%(step, accuracy, loss,data_loss,regularization_loss, self.optimizer.current_learning_rate) )

            epoch_data_loss,epoch_regularization_loss= self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss= epoch_data_loss+epoch_regularization_loss
            epoch_accuracy=self.accuracy.calculate_accumulated()
            #saving the history of training metrics
            self.history['accuracy_train'].append(epoch_accuracy)
            self.history['loss_train'].append(epoch_loss)
            
            print("training , acc: %.3f, loss: %.3f,data_loss: %.3f, reg_loss: %.3f lr: %f"%( epoch_accuracy, epoch_loss,epoch_data_loss,epoch_regularization_loss, self.optimizer.current_learning_rate) )

            if validation_data is not None :
                self.evaluate(*validation_data,batch_size=batch_size)





    def forward(self, X,training):

        self.layer_input.forward(X,training)

        for layer in self.layers:
            layer.forward(layer.prev.output,training)

        return layer.output
    
    def backward(self,output,y):
        if self.softmax_classifier_output is not None:

            self.softmax_classifier_output.backward(output,y)


            self.layers[-1].dinputs=self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output,y)

        for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)

#saving&loading parameters
    def get_params(self):
        parameters=[]

        for layer in self.trainable_layers:
            parameters.append(layer.get_params())
        return parameters


    def set_params(self,parameters):
        for parameter_set, layer in zip(parameters,self.trainable_layers):
            layer.set_params(*parameter_set)

    def save_params(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.get_params(),f)

    def load_params(self,path):
        with open(path,'rb') as f:
            self.set_params(pickle.load(f))

#saving the model

    def save(self,path):
        model= copy.deepcopy(self)

        #reset the accumulated values in loss and accuracy
        model.loss.new_pass()
        model.accuracy.new_pass()

        #we will remove inputs,output and dinput propreties from each layer

        for layer in model.layers:
            for property in ['inputs','output','dinputs','dweights','dbiases']:
                layer.__dict__.pop(property,None)

        with open(path,'wb') as f:
            pickle.dump(model,f)
# loading the model 
    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            model=pickle.load(f)

        return model

#predictions

    def predict(self,X,*,batch_size=None):

        prediction_steps=1

        if batch_size is not None:
            prediction_steps=len(X)//batch_size

            if prediction_steps* batch_size <len(X):
                prediction_steps+=1
        
        output=[]

        for step in range(prediction_steps):

            if batch_size is None :
                batch_x= X
            else:
                batch_x=X[step*batch_size:(step+1)*batch_size]
        
        batch_output= self.forward(batch_x,training=False)

        output.append(batch_output)
        return np.vstack(output)