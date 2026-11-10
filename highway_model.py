import torch
import torch.nn as nn
import torch.nn.functional as F


# https://arxiv.org/pdf/1505.00387


class HighwayLayer(nn.Module):
    def __init__(self,
            features_amount: int,
            activation_function=F.relu,
            bias: float=-3.0
        ):
        super().__init__()
        self.activation = activation_function
        # Amount of neurons has to be same as amount of features, as from paper: dimensionality of x, y, H(x, WH) and T (x, WT) must be the same for Equation (3) to be valid
        # H(x) = activation(W_H x + b_H)
        self.weights_hidden = nn.Linear( # This is affine transform, aka multiply by matrix and add vector
            features_amount,
            features_amount
        )
        # T(x) = sigmoid(W_T x + b_T)
        self.weights_transform = nn.Linear(
            features_amount,
            features_amount
        )
        # Paper suggests to initialize biases as vectors with same negative values in each cell
        # This encourages the network to initially carry information rather than transform it
        nn.init.constant_(
            self.weights_transform.bias,
            bias
        )

    def forward(self, inputs): # inputs: (samples_amount, features_amount) -> caches
        H = self.activation(self.weights_hidden(inputs)) # From paper: H is usually an affine transform (multiply by a matrix, add vector) followed by a non-linear activation function, but in general it may take other forms
        T = torch.sigmoid(self.weights_transform(inputs)) # From paper: T(x) = sigma(W^T x + b), sigma(x) = 1 / (1 + e**(-x)) https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sigmoid.html
        # From paper: y = H(x, W) * T(x, W) + x*(1 - T(x, W))
        # because there C = 1 − T
        # otherwise carry gate can be different
        return H * T + inputs * (1 - T)


class HighwayNetwork(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layers_amount: int,
        activation_function=F.relu,
        bias: int=-3
    ):
        super().__init__()
        # Plain layer: change dimensionality from input_size to hidden_size
        self.input_projection = nn.Linear( # Just affine transform
            input_size,
            hidden_size
        )
        # Highway layers: preserve dimensionality as hidden_size
        self.layers = nn.ModuleList([
            HighwayLayer(
                features_amount=hidden_size,
                activation_function=activation_function,
                bias=bias
            )
            for _ in range(layers_amount)
        ])
        # Plain layer: change dimensionality from hidden_size to output_size
        self.output_projection = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, inputs):
        current = self.input_projection(inputs)
        for layer in self.layers:
            current = layer(current)
        # From paper: In cases when it is desirable to change the size of the representation, one can replace x with ˆx obtained by suitably sub-sampling or zero-padding x. Another alternative is to use a plain layer (without highways) to change dimensionality and then continue with stacking highway layers
        final = self.output_projection(current)
        return final
