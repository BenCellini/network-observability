# network-observability
Evaluate the observability of artificial neural network weights and biases from arbitrary measured outputs. Also has code to visualize feedforward network structures in FCNN style.

This repo relies heavily on my other project for empirical observability:
https://github.com/vanbreugel-lab/pybounds

# Examples
Start here for a basic example on a pytorch model.
[network_observability_example.ipynb](notebooks%2Fnetwork_observability_example.ipynb)

This example creates a simple feedforward PyTorch model with a linear output functions and randomized weights (not trained for a specific purpose). Then, it uses random sets of inputs to construct an observability matrix given measurements from specified neurons (output or hidden layer neurons). Then the Fisher information + inverse is computed and used to assess the observability of each network weight/bias. The network is visualized such that the measured neurons are indicated and the connections are colored by their observability level.
