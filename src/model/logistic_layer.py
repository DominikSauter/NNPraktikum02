import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    n_in: int: number of units from the previous layer (or input data)
    n_out: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    is_classifier_layer: bool:  to do classification or regression

    Attributes
    ----------
    n_in : positive int:
        number of units from the previous layer
    n_out : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activation_string : string
        the name of the activation function
    is_classifier_layer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, n_in, n_out, weights=None,
                 activation='sigmoid', is_classifier_layer=False):

        # Get activation function from string
        self.activation_string = activation
        self.activation = Activation.get_activation(self.activation_string)

        self.n_in = n_in
        self.n_out = n_out

        #self.inp = np.ndarray((n_in+1, 1))
        self.inp = np.ndarray((n_in+1))
        self.inp[0] = 1
        #self.outp = np.ndarray((n_out, 1))
        self.outp = np.ndarray((n_out))
        #self.deltas = np.zeros((n_out, 1))
        self.deltas = np.zeros((n_out))

        # You can have better initialization here
        if weights is None:
            self.weights = np.random.rand(n_in+1, n_out)/10
        else:
            self.weights = weights

        #self.is_classifier_layer = is_classifier_layer

        # Some handy properties of the layers
        #self.size = self.n_out
        #self.shape = self.weights.shape


    def forward(self, inp, classify):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy array (1,n_in + 1) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (1,n_out) containing the output of the layer
        """

        # Here you have to implement the forward pass
        
        # create input numpy.ndarray: first element = 1 (constant for bias), rest is inp
        input = np.concatenate((np.array([self.inp[0]]), np.array(inp)), axis=0)
        # allocate output numpy.ndarray
        output = np.ndarray((self.n_out))
        # for each neuron in the current layer: compute output
        for i in xrange(self.n_out):
            output[i] = self._fire(input, self.weights[:,i])    # self.weights[:,i] - get specific column of weight matrix for the current neuron
  
        # if we are training and not classifying: update inp and outp
        if (not classify):
            self.inp = input
            self.outp = output
            
        return output


    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        # Here the implementation of partial derivative calculation
       
        # if this is the output layer (i.e. there is no next layer with weights):
        # computed deltas are passed via nextDerivatives
        if nextWeights is None:
            self.deltas = nextDerivatives
        # else this is a hidden layer
        else:
            # for all neurons in this layer
            for m in xrange(self.n_out):
                sigmoidTerm = self.outp[m] * (1 - self.outp[m])
                downstreamSum = 0
                # for all neurons in the next layer
                for n in xrange(len(nextDerivatives)):
                    downstreamSum += nextDerivatives[n] * nextWeights[m+1,n]    # m+1 because the first weight is the bias
                self.deltas[m] = sigmoidTerm * downstreamSum


    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """

        # Here the implementation of weight updating mechanism
        
        # for all neurons in this layer
        for m in xrange(self.n_out):
            # for all inputs/weights of one neuron
            for n in xrange(len(self.inp)):
                self.weights[n,m] += learningRate * self.deltas[m] * self.inp[n]


    def _fire(self, inp, weightsOfNeuron):
        return Activation.sigmoid(np.dot(np.array(inp), np.array(weightsOfNeuron)))
