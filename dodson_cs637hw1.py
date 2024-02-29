"""
Mitchell Dodson
CS 637 HW 1

Implementation of backpropagation using the following basic neural network:

x1
  \
   w1
    \
     sum - logistic
    /              \
   w2               \
  /                  w5
x2                    \
                       \
                        sum - logistic - y
                       /
x3                    /
  \                  w6
   w3               /
    \              /
     sum - logistic
    /
   w4
  /
x4

To evaluate the gradients of the loss function in terms of the weights and
inputs provided in the assignment, just run this script directly using the
values hard-coded in the __main__ context below. Otherwise modify the T (truth)
W (weights) and/or X (inputs) values below to change the structure.
"""
import numpy as np
from typing import Callable

def sigmoid(x):
    """ Logistic function """
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    """ Derivative of logistic function """
    s = sigmoid(x)
    return s*(1-s)

def l2_loss(pred, true):
    """ L2 loss function """
    return (pred-true)**2

def d_l2_loss(pred, true):
    """ Derivative of L2 loss function """
    return 2*np.abs(pred-true)

class LinearAndLogisticLayer:
    """
    Encodes forward and backward passes for a basic neural network layer
    consisting of a linear equation of the inputs followed by a logistic
    activation function (which is optionally replaceable)
    """
    def __init__(self, weights:np.array, activation:Callable=sigmoid,
                 d_activation:Callable=d_sigmoid):
        """
        :@param weights: 1D numpy array of initial weight values for this layer
        :@param activation: Function providing the scalar activation function
        :@param d_activation: Function providing the activation derivative
        """
        self._w = np.asarray(weights)
        self._f = activation
        self._df = d_activation
        ## Most recent inputs
        self._prev_in = None
        ## Activation function output for most recent inputs
        self._prev_out = None
        ## Derivative of activation function at most recent inputs
        self._prev_der = None

    def forward(self, inputs:np.array):
        """
        Evaluate the output of the layer given the inputs.

        This method stores as attributes the user-provided inputs, the layer
        output, and the derivative of the layer activation at the input values.

        :@param inputs: 1D numpy array with the same size as the layer weights.
        :@return: Layer output as a scalar value
        """
        inputs = np.asarray(inputs)
        assert inputs.shape == self._w.shape
        self._prev_in = inputs
        ## Perform linear transform
        z = np.sum(inputs * self._w)
        ## Save the output and derivative states for the  provided input
        self._prev_out = self._f(z)
        self._prev_der = self._df(z)
        return self._prev_out

    def backward(self, loss_grad:float, include_weight_grads=False):
        """
        Evaluate the gradient of the divergence (loss) function with respect to
        the layer inputs so that it can be propagated to prior layers,
        optionally including the gradient of this layer's weights.

        Critically, the forward pass using inputs corresponding to this
        backward pass must be executed immediately prior to this method in
        order for the required attributes to be retained.

        This method also relies on that the forward pass uses a linear
        transform to reduce the inputs with the weights.

        :@param loss_grad: Gradient of the divergence function with respect to
            this layer's output .
        :@param include_weight_grads: If True, weight gradients are returned as
            well as input gradients as a 2-tuple (grad_inputs, grad_weights)
        :@return: Gradient of divergence function wrt layer inputs as a 1D
            array with the same size as the weights, or a 2-tuple of array if
            include_weight_grads is True.
        """
        ## derivative of loss wrt pre-activation values (scalar)
        dloss_dz = loss_grad * self._prev_der
        ## partial derivatives of loss wrt inputs (1D)
        dloss_dx = dloss_dz * self._w
        if include_weight_grads:
            ## partial derivatives of loss wrt weights (1D)
            dloss_dw = dloss_dz * self._prev_in
            return dloss_dx, dloss_dw
        return dloss_dx


if __name__=="__main__":
    T = 0.5 ## True value
    W = (-1.7, 0.1, -0.6, -1.8, -0.2, 0.5) ## Weights
    X = (-0.7, 1.2, 1.1, -2) ## Inputs

    ## Layer declarations
    L1 = LinearAndLogisticLayer(weights=W[0:2])
    L2 = LinearAndLogisticLayer(weights=W[2:4])
    L3 = LinearAndLogisticLayer(weights=W[4:6])

    ## Forward pass
    h1 = L1.forward(inputs=X[0:2])
    h2 = L2.forward(inputs=X[2:4])
    P = L3.forward(inputs=(h1,h2)) ## Prediction

    ## Divergence
    div = l2_loss(P, T)
    ddiv_dp = d_l2_loss(P, T)

    print("\n --( Forward pass results )-- \n")
    print(f"Prediction : {P}")
    print(f"Divergence : {div}")

    ## Backward Pass
    kw = {"include_weight_grads":True}
    ddiv_dx_l3, ddiv_dw_l3 = L3.backward(ddiv_dp, **kw)
    ddiv_dx_l2, ddiv_dw_l2 = L2.backward(ddiv_dx_l3[1], **kw)
    ddiv_dx_l1, ddiv_dw_l1 = L1.backward(ddiv_dx_l3[0], **kw)

    ## Results
    print("\n --( Loss gradient wrt weights )-- \n")
    ddiv_dws = (*ddiv_dw_l1, *ddiv_dw_l2, *ddiv_dw_l3)
    ddiv_dxs = (*ddiv_dx_l1, *ddiv_dx_l2) ## exclude hidden layer inputs
    results = [f"dDiv/dW_{i+1}  : {ddiv_dws[i]:.4g}"
               for i in range(len(ddiv_dws))]
    print("\n".join(results))
    print("\n --( Loss gradient wrt inputs )-- \n")
    ddiv_dws = (*ddiv_dw_l1, *ddiv_dw_l2, *ddiv_dw_l3)
    results = [f"dDiv/dX_{i+1}  : {ddiv_dxs[i]:.4g}"
               for i in range(len(ddiv_dxs))]
    print("\n".join(results))
