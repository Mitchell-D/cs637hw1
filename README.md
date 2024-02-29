# cs637hw1

Custom backpropagation algorithm for a basic neural network

### Dependencies:

 - `numpy`
 - `python>=3.9`

```
x1
  \
   w1
    \
     sum -- logistic -- h1
    /                     \
   w2                      \
  /                         w5
x2                           \
                              \
                               sum -- logistic -- y
                              /
x3                           /
  \                         w6
   w3                      /
    \                     /
     sum -- logistic -- h2
    /
   w4
  /
x4
```

To evaluate the gradients of the loss function in terms of weights
and inputs from the assignment, run `python dodson_cs637hw1.py`
using the values hard-coded in the `__main__` context.

Otherwise, to customize the layer widths or parameter values, modify
the T (truth) W (weights) and/or X (inputs) variables in `__main__`
