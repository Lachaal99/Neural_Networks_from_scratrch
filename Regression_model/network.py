import numpy as np

# defining the class layer where each layer of our network will be from this class 
class layer_dense: 

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1= 0,bias_regularizer_l1 =0,weight_regularizer_l2 =0,bias_regularizer_l2 =0): 
        self.weights= 0.01* np.random.randn(n_inputs,n_neurons)
        
        self.biases= np.zeros((1,n_neurons))

        self.weight_regularizer_l1=weight_regularizer_l1
        self.bias_regularizer_l1= bias_regularizer_l1
        self.weight_regularizer_l2= weight_regularizer_l2
        self.bias_regularizer_l2= bias_regularizer_l2

    def forward(self, input): 
        self.inputs= input
        
        self.output= np.dot(input, self.weights) + self.biases
    
    def backward(self, dvalues): 
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbiases= np.sum(dvalues,axis=0, keepdims=True)

        self.dinputs= np.dot(dvalues, self.weights.T)

        # L1 and L2 regularization expressins in backpropagation
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

# define the dropout layers for our training 
class Layer_Dropout: 

    def __init__(self, rate):
        self.rate = 1-rate

    def forward(self,inputs): 
        self.inputs= inputs

        self.binary_mask= np.random.binomial(1, self.rate, size=inputs.shape) /self.rate
        self.output = inputs* self.binary_mask

    def backward(self,dvalues): 
        self.dinputs= dvalues*self.binary_mask    








#activation functions

# ReLU activation functiono
class Activation_ReLu: 
    def forward(self, inputs):

        self.inputs= inputs

        self.output = np.maximum(0,inputs)
        
    
    def backward(self,dvalues):
        self.dinputs= dvalues.copy()


        self.dinputs[self.inputs<=0]=0

# Softmax activation 

class Activation_SoftMax: 
    def forward(self ,inputs):
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

# Activation sigmoid
class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs= inputs
        self.output= 1/(1+np.exp(-inputs))
   
    def backward(self,dvalues):
        self.dinputs= dvalues*(1-self.output)*self.output  

# activation linear

class Activation_Linear:

    def forward(self,inputs):

        self.inputs= inputs
        self.output= inputs

    def backward(self,dvalues):
        self.dinputs= dvalues.copy()


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
    def __init__(self, learning_rate = 0.07, decay= 0., epsilon= 1e-7, beta_1=0.9, beta_2=0.999):
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
    def regularization_loss( self, layer): 
        regularization_loss= 0

        if layer.weight_regularizer_l1 > 0: 
            regularization_loss += layer.weight_regularizer_l1 *np.sum(np.abs(layer.weights))  

        if layer.bias_regularizer_l1 > 0: 
            regularization_loss += layer.bias_regularizer_l1 *np.sum(np.abs(layer.biases))

        if layer.weight_regularizer_l2 > 0: 
            regularization_loss += layer.weight_regularizer_l2 *np.sum(layer.weights* layer.weights)

        if layer.bias_regularizer_l2 > 0: 
            regularization_loss += layer.bias_regularizer_l2 *np.sum(layer.biases* layer.biases)    

        return regularization_loss


    def calculate(self,output,y): 
        sample_loss= self.forward(output,y)
        data_loss= np.mean(sample_loss)
        return data_loss  
      
    
class Loss_CategoricalCrossEntropy(Loss): 



    
    def forward(self, y_true, y_pred): 
        samples= len(y_pred)
        
        y_pred_clipped= np.clip(y_pred,1e-7,1 - 1e-7)

        if (len(y_true.shape)==1):
            correct_confidences = y_pred_clipped[range(samples),y_true]
        
        elif (len(y_true.shape)==2): 
            correct_confidences= np.sum(y_true* y_pred_clipped, axis=1)    
        
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
        
