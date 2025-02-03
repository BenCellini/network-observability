
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings


class FCNNVisualizer:
    def __init__(self, layers=None, layer_space=1.5,
                 neuron_radius=0.3, neuron_edgewidth=1.5,
                 connection_linewidth=2.0, connection_alpha=0.9):
        """
        Initializes the FCNNVisualizer with layer sizes and default colors.

        param: iterable|None of ints layers:  each integer represents the number of neurons in that laye
        param: float neuron_radius: radius of the neurons
        param: float layer_space: spacing between layers
        param: float connection_alpha: connection alpha
        """

        # Set layer sizes
        if layers is None:
            self.layers = (8, 6, 4, 2, 4, 6, 8)
        else:
            self.layers = layers

        self.n_layer = len(self.layers)

        # Calculate number of neurons & connections
        self.n_neuron = sum(self.layers)
        self.n_neuron_edge = self.n_neuron
        self.n_connection = sum(a * b for a, b in zip(self.layers, self.layers[1:]))

        # Set drawing properties
        self.neuron_radius = neuron_radius
        self.connection_alpha = connection_alpha
        self.layer_width = layer_space  # horizontal spacing between layers
        self.neuron_positions = []
        self.neuron_edgewidth = neuron_edgewidth
        self.connection_linewidth = connection_linewidth
        self.neuron_colors = None
        self.neuron_edge_colors = None
        self.connection_colors = None
        self.layer_continuation = np.zeros(self.n_neuron, dtype=bool)

        self.colormaps = {
            'neuron': mpl.colors.LinearSegmentedColormap.from_list('b_g_r', ['whitesmoke', 'gray', 'gold']),
            'connection': plt.get_cmap('inferno')}

        # Set default neuron colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('b_g_r', ['grey', 'grey'])
        self.set_colors(name='neuron',
                        values=np.linspace(0.0, 1.0, num=self.n_neuron).tolist(),
                        colormap=cmap)

        # Set default neuron edge colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('b_g_r', ['black', 'black'])
        self.set_colors(name='neuron_edge',
                        values=np.linspace(0.0, 1.0, num=self.n_neuron).tolist(),
                        colormap=cmap)

        # Set default connection colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('b_g_r', ['darkgrey', 'darkgrey'])
        self.set_colors(name='connection',
                        values=np.linspace(0.0, 1.0, num=self.n_connection).tolist(),
                        colormap=cmap)

    def set_colors(self, name: str, colors: list[float] = None, values: list[float] = None, colormap=None, norm=None):
        """
        Sets a class property (specified by name) based on a list of colors or values and a colormap.

        param str name: name of the color property to set, "neuron" or "connection"
        param float|str|list[float|str] colors: iterable f RGB values or strings. Can not be set if values is not None.
        param list[float] values: numerical values to map to colors. Can not be set if colors is not None.
        param str | matplotlib.colors.Colormap colormap: name of a Matplotlib colormap or an instance of a Colormap
        param matplotlib.colors.Normalize norm: Matplotlib normalization object to scale values.
        If None, automatically created based on `values`.

        Normalization and colormap only have an effect if values is not None.

        raises AttributeError: If the class does not have a property with the specified `name`.
        """

        # Make sure colors are set based on colors or values variable
        if (colors is None) & (values is None):
            raise Exception('Either colors or values must be set.')
        elif (colors is not None) & (values is not None):
            raise Exception('Only one of colors or values can be set.')
        else:
            pass

        # Set the property dynamically
        name_values = name + '_colors'
        N = getattr(self, 'n_' + name)  # number of colors to set
        if not hasattr(self, name_values):
            raise AttributeError(f"Class does not have a property named '{name_values}'.")

        # Set colors
        if values is not None:  # if color is set based on value
            if len(values) != N:
                raise Exception(f"Number of values in values must be equal to the number of' {name}s.")

            # Set colormap
            if colormap is None:
                cmap = self.colormaps[name]
            else:
                if isinstance(colormap, str):
                    cmap = plt.get_cmap(colormap)
                else:
                    cmap = colormap

            # Set the color normalization
            if norm is None:
                norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values), clip=False)

            # Set the colors from the colormap with normalization
            colors_rgb = [cmap(norm(v)) for v in values]

        else:  # set colors based on input colors
            try:  # check if colors are specified as a literal or a single value
                _ = iter(colors)
            except TypeError:  # string or not iterable, set all colors to the value in colors
                colors_rgb = [colors for _ in range(N)]
            else:  # string or other iterable, set colors directly
                if isinstance(colors, str):  # string given, set for all colors
                    colors_rgb = [colors for _ in range(N)]
                elif len(colors) != N:  # other iterable given, but not correct length
                    raise Exception(f"Number of values in values must be equal to the number of' {name}s.")
                else:  # iterable with colors in correct format
                    colors_rgb = colors

        # Set the class color attribute
        setattr(self, name_values, colors_rgb)

    def calculate_neuron_positions(self):
        """Calculates and stores the positions of neurons in each layer."""
        self.neuron_positions = []
        for layer_idx, n_neurons in enumerate(self.layers):
            # Center neurons vertically for each layer
            y_positions = np.linspace(-0.5 * (n_neurons - 1), 0.5 * (n_neurons - 1), n_neurons)
            y_positions = np.flip(y_positions)
            x_position = layer_idx * self.layer_width
            layer_positions = [(x_position, y) for y in y_positions]
            self.neuron_positions.append(layer_positions)

    def set_layer_continuation(self, neuron_index='center', no_ends=False, params=None):
        """ Specify neurons to replace with continuation dots.
        """

        # Set neuron mask
        neuron_index_list = self.layer_continuation.copy()
        if neuron_index == 'center':
            if params is None:
                params = {'layers': range(0, self.n_layer)}

            # for n, layer_size in enumerate(self.layers):
            for n in params['layers']:
                layer_size = self.layers[n]

                # Get start index of current layer
                start_layer_index = int(sum(self.layers[0:n]))

                # Find halfway point in layer
                layer_index = int(np.floor(layer_size/2))
                if layer_index == (layer_size/2):
                    print("warning: layer #" + str(n + 1) + " size should be odd for centered continuation dots.")

                # Set the neuron index to replace with continuation dots
                neuron_index = start_layer_index + layer_index
                neuron_index_list[neuron_index] = True

        elif neuron_index == 'pattern':
            if params is None:
                params = {'start': 1, 'on': 2, 'off': 3, 'layers': (0,)}

            for n in params['layers']:
                start_layer_index = int(sum(self.layers[0:n]))
                sequence = create_parametrized_sequence(start=params['start'],
                                                        odd_spacing=params['on'] - 1,
                                                        even_spacing=params['off'] + 1,
                                                        length=self.layers[n])

                sequence = sequence[sequence < self.layers[n]]
                print(sequence)
                for j in sequence:
                    neuron_index = start_layer_index + j
                    neuron_index_list[neuron_index] = True

        else:
            raise ValueError('neuron_index option not available.')

        # Update the layer continuation mask
        for n, layer_size in enumerate(self.layers):
            start_layer_index = int(sum(self.layers[0:n]))
            for j in np.arange(0, self.layers[n], step=1):
                neuron_index = start_layer_index + j
                if no_ends:
                    if (n <= 0) or (n >= (self.n_layer - 1)):
                        pass
                    else:
                        self.layer_continuation[neuron_index] = neuron_index_list[neuron_index]
                else:
                    self.layer_continuation[neuron_index] = neuron_index_list[neuron_index]

    def draw(self, ax=None, dpi=100):
        """Draws the fully connected feedforward neural network.
        """
        self.calculate_neuron_positions()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
            ax.axis('off')

        # Draw connections between neurons in adjacent layers first
        connection_idx = 0
        for layer_idx in range(len(self.layers) - 1):
            layer_a = self.neuron_positions[layer_idx]
            layer_b = self.neuron_positions[layer_idx + 1]
            neuron_idx_a = int(sum(self.layers[:layer_idx]))
            neuron_idx_b = int(sum(self.layers[:layer_idx + 1]))

            for (i, (x_a, y_a)) in enumerate(layer_a):
                from_cont = self.layer_continuation[neuron_idx_a + i]  # Check source neuron

                for (j, (x_b, y_b)) in enumerate(layer_b):
                    to_cont = self.layer_continuation[neuron_idx_b + j]  # Check target neuron

                    # Block connection if either the source or target neuron should not continue
                    if not (from_cont or to_cont):
                        connection_alpha = self.connection_alpha
                    else:
                        connection_alpha = 0.0

                    connection_color = self.connection_colors[connection_idx]
                    ax.plot([x_a, x_b], [y_a, y_b],
                            color=connection_color,
                            alpha=connection_alpha,
                            linewidth=self.connection_linewidth,
                            zorder=1)

                    connection_idx += 1

        # Draw neurons on top of connections
        neuron_idx = 0
        for layer_positions in self.neuron_positions:
            # Draw neuron
            for (x, y) in layer_positions:
                if not self.layer_continuation[neuron_idx]:  # draw a regular neuron
                    neuron_color = self.neuron_colors[neuron_idx] if self.neuron_colors else 'gray'
                    neuron_edge_color = self.neuron_edge_colors[neuron_idx] if self.neuron_colors else 'black'
                    circle = plt.Circle((x, y), self.neuron_radius,
                                        color=neuron_color,
                                        ec=neuron_edge_color,
                                        lw=self.neuron_edgewidth)
                    ax.add_patch(circle)
                else:  # draw continuation dots
                    draw_vertical_circles(radius=self.neuron_radius/5,
                                          spacing=self.neuron_radius,
                                          center_position=(x, y),
                                          num_circles=3,
                                          color='black',
                                          ax=ax)

                neuron_idx += 1

        # Set axis limits and display
        max_neurons = max(self.layers)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, len(self.layers) * self.layer_width - 0.5)
        ax.set_ylim(-0.5 * max_neurons, 0.5 * max_neurons)


def draw_vertical_circles(ax=None, radius=1.0, spacing=2.0, center_position=(0.0, 0.0), num_circles=3, color='black'):
    """ Draws vertically aligned circles.

    :param ax:
    :param radius:
    :param spacing:
    :param center_position:
    :param num_circles:
    :param color:
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Unpack center position
    center_x, center_y = center_position

    # Calculate y-positions for the circles
    start_y = center_y + spacing * (num_circles - 1) / 2
    positions_y = [start_y - i * spacing for i in range(num_circles)]

    # Draw circles
    for y in positions_y:
        circle = patches.Circle((center_x, y), radius, color=color, fill=True)
        ax.add_patch(circle)

    # Set equal aspect ratio and adjust plot limits
    ax.set_aspect('equal')
    buffer = radius + spacing
    ax.set_xlim(center_x - buffer, center_x + buffer)
    ax.set_ylim(center_y - buffer - spacing * (num_circles - 1) / 2,
                center_y + buffer + spacing * (num_circles - 1) / 2)


def create_parametrized_sequence(start=0, length=10, odd_spacing=3, even_spacing=1):
    seq = [start]
    for i in range(1, length):
        if i % 2 == 1:  # odd index
            seq.append(seq[-1] + odd_spacing)
        else:  # even index
            seq.append(seq[-1] + even_spacing)
    return np.array(seq)
