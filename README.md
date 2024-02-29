# cs637hw1

Custom backpropagation algorithm for a basic neural network

```
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
```

To evaluate the gradients of the loss function in terms of the
weights and inputs provided in the assignment, just run this script
directly using the values hard-coded in the `__main__` context.
Otherwise modify the T (truth) W (weights) and/or X (inputs) values
to change the structure.
