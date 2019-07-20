
# Interpreting neural nets by training them to remove nonlinearities

This repo contains a short paper and sample code demonstrating
a simple solution that makes it possible to
extract rules from a neural network that employs Parametric Rectified Linear Units (PReLUs).
We introduce a force, applied in parallel to backpropagation, that
aims to reduce PReLUs into the identity function, which then causes
the neural network to collapse into a smaller system of linear functions and inequalities
suitable for review or use by human decision makers.

As this force reduces the capacity of neural networks, it is expected to help avoid overfitting as well.
