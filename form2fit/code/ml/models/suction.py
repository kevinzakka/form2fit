import torch

from form2fit.code.ml.models.base import BaseModel
from form2fit.code.ml.models.fcn import FCNet


class SuctionNet(BaseModel):
    """The suction prediction network.

    Attributes:
        num_channels: (int) The number of channels in the
            input tensor.
    """

    def __init__(self, num_channels):
        super().__init__()

        self.num_channels = num_channels
        self._fcn = FCNet(num_channels, 1)

    def forward(self, x):
        return self._fcn(x)