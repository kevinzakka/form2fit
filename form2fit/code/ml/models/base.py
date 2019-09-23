"""Abstract model class.
"""

import os

import torch


class BaseModel(torch.nn.Module):
    """An abstract base class for a deep neural network.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError

    def load_weights(self, filename, device):
        """Loads the weights of a model from a given directory.
        """
        state_dict = torch.load(filename, map_location=device)
        try:
            self.load_state_dict(state_dict)
            print("Successfully loaded model weights from {}.".format(filename))
        except:
            print("[!] Could not load model weights. Training from scratch instead.")

    def save_weights(self, filename):
        """Saves the weights of a model to a given directory.
        """
        torch.save(self.state_dict(), filename)

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters())
