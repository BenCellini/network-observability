import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class NetworkSimulator:
    def __init__(self, model: nn.Module, device: torch.device, output_mode='full'):
        """
        Initializes the NetworkSimulator with a PyTorch model, captures the size
        of each layer's weights, and stores the initial weights in three formats:
        as a single concatenated tensor, as a NumPy array, and as a list of tensors
        per layer. It also sets up hooks to capture the output of every layer during
        the simulation.

        :param model: PyTorch neural network model to be used in the simulation
        :type model: nn.Module

        :param device: PyTorch device to be used in the simulation
        :type model: torch.device

        :param output_mode: output the final layer outputs only ('final') or the
                            full output from every hidden layer & final layer ('full')

        """
        self.output_mode = output_mode

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Set model
        self.model = model.to(self.device)

        # Layer sizes
        self.layer_info = get_model_layers_info(self.model)
        self.layer_sizes = tuple(self.layer_info.values())
        self.layer_names = tuple(self.layer_info.keys())

        # Precompute the size (shape) and number of elements for each parameter in the model
        self.param_shapes = [param.shape for param in self.model.parameters()]
        self.param_sizes = [param.numel() for param in self.model.parameters()]
        self.final_layer_size = tuple(self.param_shapes[-1])[0]  # size of final layer

        # Store initial weights in three formats:
        # - A single concatenated tensor
        self.initial_weights_tensor = torch.cat([param.data.view(-1) for param in self.model.parameters()])
        # - A NumPy array
        self.initial_weights_numpy = self.initial_weights_tensor.detach().cpu().numpy()
        # - A list of tensors per layer
        self.initial_weights_per_layer = [param.data.clone() for param in self.model.parameters()]

        # Set layer weight/bias names
        self.layer_labels = []
        self.weight_labels = []
        for name, param in self.model.named_parameters():
            layer_label = str(name).replace('.', '_')
            self.layer_labels.append(layer_label)

            # Create label for every weight/bias in each layer
            layer_weights = param.flatten()
            layer_weights_size = layer_weights.shape[0]
            for w in range(layer_weights_size):
                weight_label = layer_label + '_' + str(w)
                self.weight_labels.append(weight_label)

        # Set weight names as state names for observability analysis
        self.state_names = self.weight_labels.copy()

        # Set up hooks to capture the outputs of every layer
        self.layer_outputs = []
        self.hooks = []
        self._register_hooks()

        # Set output
        self.output_labels = []
        self.measurement_names = []

        # Run model once
        self.output_mask = None
        u0 = np.random.normal(loc=1.0, scale=1.0, size=self.layer_sizes[0])
        self.y = self.simulate(x0=None, u=u0)

        # Set mask to determine which outputs are used
        self.output_mask = np.ones_like(self.y, dtype='bool').squeeze()

    def _register_hooks(self):
        """ Registers forward hooks to capture the output of every layer in the model.
        """

        def hook_fn(module, input, output):
            self.layer_outputs.append(output)  # store the output of each layer

        # Register hooks for every layer in the model
        for layer in self.model.children():
            hook = layer.register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def set_output_labels(self):
        """ Sets the output labels of the model.
        """

        n_layer = len(self.layer_outputs)
        self.output_labels = []
        for layer_index, layer_output in enumerate(self.layer_outputs):
            # Flatten the layer output so each neuron output can be labeled individually
            flat_output = layer_output.view(-1)  # convert to a flat tensor to access each scalar

            # Generate a label for each neuron in the current layer
            for neuron_index in range(flat_output.numel()):
                lab = f"{self.layer_names[layer_index + 1]}_neuron_{neuron_index + 1}"
                # Only append final layer if output mode is 'final'
                if self.output_mode == 'final':
                    if layer_index == (n_layer - 1):
                        self.output_labels.append(lab)

                elif self.output_mode == 'full':  # append every layer
                    self.output_labels.append(lab)

        # Set measurement names based on output mask
        if self.output_mask is not None:
            self.measurement_names = []
            for n, name in enumerate(self.output_labels):
                if self.output_mask[n]:
                    self.measurement_names.append(name)
        else:
            self.measurement_names = self.output_labels.copy()

    def set_weights(self, x):
        """
        Sets the weights of the model. The weights can be provided as a single
        flat list, numpy array, or tensor, where all weights are sequentially
        ordered, or as a list of tensors for each layer.

        :param x: The weights for the model. Can be a single list, numpy array,
                  or tensor with all weights concatenated, or a list where each
                  element corresponds to the weights for one layer.
        :type x: list, np.ndarray, or torch.Tensor
        """
        # Convert x to a torch.Tensor if it is not already
        if isinstance(x, (list, np.ndarray)):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():  # disable gradient tracking
            start_idx = 0
            for i, param in enumerate(self.model.parameters()):
                # Slice and reshape x based on precomputed shapes and sizes
                num_elements = self.param_sizes[i]
                param_shape = self.param_shapes[i]
                param.data = x[start_idx:start_idx + num_elements].view(param_shape)
                start_idx += num_elements

    def simulate(self, x0=None, u=None, numpy_output=True):
        """
        Simulates the model with the given input and weights.

        :param x0: the weights for the model. Can be a single list, numpy array,
                  or tensor with all weights concatenated, or a list where each
                  element corresponds to the weights for one layer. If None,
                  the model's current weights are used (typically the initial weights).
        :type x0: list, np.ndarray, or torch.Tensor, optional

        :param u: The input tensor to the network over time (w x m) where w is the number
                  of time-steps & b is the number of inputs
        :type u: torch.Tensor, np.ndarray

        :return y: The output of the network after performing a forward pass with the specified weights.
        :rtype y: torch.Tensor, np.ndarray

        :param numpy_output: boolean to mke the output y a numpy array

        """

        # Only set the model's weights if x is provided
        if x0 is not None:
            self.set_weights(x0)

        # Convert u to a torch.Tensor if it is not already
        if isinstance(u, (list, np.ndarray)):
            u = torch.tensor(u, dtype=torch.float32).to(self.device)

        # Make sure inputs are at least 2D
        u = torch.atleast_2d(u)

        # Run the model forward pass with input u
        with torch.no_grad():  # Disable gradient tracking for inference
            w = u.shape[0]
            y = []
            for k in range(w):
                # Clear previous layer outputs
                self.layer_outputs = []

                # Get inputs at current time-step
                u_k = u[k, :]

                # Run the model and get the outputs (ony of final layer)
                y_k = self.model(u_k)

                # Get all the outputs from each hidden layer and final layer
                y_k_full = torch.hstack(self.layer_outputs)

                # Set the outputs
                if self.output_mode == 'full':
                    y.append(y_k_full)
                elif self.output_mode == 'final':
                    y.append(y_k)

        # Stack the outputs over time
        y = torch.vstack(y)
        y = torch.atleast_2d(y)

        # Set what outputs to use
        if self.output_mask is not None:
            y = y[:, self.output_mask]

        # Convert to numpy array
        if numpy_output:
            y = y.cpu().data.numpy()

        # Set the output labels
        self.set_output_labels()

        # Store output
        self.y = pd.DataFrame(y, columns=self.measurement_names)

        return y

    def get_layer_outputs(self):
        """
        Returns the outputs of all layers after the most recent simulation.

        :return: A list of outputs for each layer in the network.
        :rtype: list of torch.Tensor
        """
        return self.layer_outputs


def get_model_layers_info(model):
    layer_sizes = []
    layer_names = {}

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            # For the first layer, capture the input size
            if not layer_sizes:
                layer_sizes.append(layer.in_features)
                layer_names["input"] = layer.in_features

            layer_names[name] = layer.out_features

        elif isinstance(layer, nn.Conv2d):
            # For the first layer, capture the input channels
            if not layer_sizes:
                layer_sizes.append(layer.in_channels)
                layer_names["input"] = layer.in_channels

            layer_names[name] = layer.out_channels

    return layer_names


class FCNNVisualizer:
    def __init__(self, layers):
        """
        Initializes the FCNNVisualizer with layer sizes and default colors.

        Parameters:
            layers (list of int): List where each integer represents the number of neurons in that layer.
        """
        self.layers = layers
        self.neuron_colors = None
        self.connection_colors = None
        self.neuron_radius = 0.3
        self.connection_alpha = 0.9
        self.layer_width = 1.5  # Horizontal spacing between layers
        self.neuron_positions = []

        self.colormaps = {
            'neuron': mpl.colors.LinearSegmentedColormap.from_list('b_g_r', ['whitesmoke', 'gray', 'gold']),
            'connection': plt.get_cmap('inferno')}

    def set_colors(self, name: str, values: list[float] = None, colormap=None, norm=None):
        """
        Sets a class property (specified by name) based on a list of values and a colormap.

        param name: name of the property to set
        type name: str
        param values: List of numerical values to map to colors.
        type values: list[float]
        param colormap: Name of a Matplotlib colormap or an instance of a Colormap. If None, defaults to 'weight' in `self.colormaps` or 'viridis'.
        type colormap: str | matplotlib.colors.Colormap
        param norm: A Matplotlib normalization object to scale values to [0, 1]. If None, it is automatically created based on `values`.
        type norm: matplotlib.colors.Normalize

        raises AttributeError: If the class does not have a property with the specified `name`.
        """
        if colormap is None:
            cmap = self.colormaps[name]
        else:
            if isinstance(colormap, str):
                cmap = plt.get_cmap(colormap)
            else:
                cmap = colormap

        if norm is None:
            norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values), clip=False)

        # Set the property dynamically
        name_values = name + '_colors'
        if not hasattr(self, name_values):
            raise AttributeError(f"Class does not have a property named '{name_values}'.")

        setattr(self, name_values, [cmap(norm(v)) for v in values])

    def calculate_neuron_positions(self):
        """Calculates and stores the positions of neurons in each layer."""
        self.neuron_positions = []
        for layer_idx, n_neurons in enumerate(self.layers):
            # Center neurons vertically for each layer
            y_positions = np.linspace(-0.5 * (n_neurons - 1), 0.5 * (n_neurons - 1), n_neurons)
            x_position = layer_idx * self.layer_width
            layer_positions = [(x_position, y) for y in y_positions]
            self.neuron_positions.append(layer_positions)

    def draw(self):
        """Draws the fully connected feedforward neural network."""
        self.calculate_neuron_positions()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Draw connections between neurons in adjacent layers first
        connection_idx = 0
        for layer_idx in range(len(self.layers) - 1):
            layer_a = self.neuron_positions[layer_idx]
            layer_b = self.neuron_positions[layer_idx + 1]

            for (x_a, y_a) in layer_a:
                for (x_b, y_b) in layer_b:
                    connection_color = self.connection_colors[connection_idx] if self.connection_colors else 'gray'
                    ax.plot([x_a, x_b], [y_a, y_b], color=connection_color, alpha=self.connection_alpha, linewidth=2,
                            zorder=1)
                    connection_idx += 1

        # Draw neurons on top of connections
        neuron_idx = 0
        for layer_positions in self.neuron_positions:
            for (x, y) in layer_positions:
                neuron_color = self.neuron_colors[neuron_idx] if self.neuron_colors else 'black'
                circle = plt.Circle((x, y), self.neuron_radius, color=neuron_color, ec='black', lw=1.5)
                ax.add_patch(circle)
                neuron_idx += 1

        # Set axis limits and display
        max_neurons = max(self.layers)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, len(self.layers) * self.layer_width - 0.5)
        ax.set_ylim(-0.5 * max_neurons, 0.5 * max_neurons)
