import torch.multiprocessing as mp 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn import Dropout, Conv3d, BatchNorm3d, MaxPool3d, AdaptiveAvgPool3d
from typing import Callable, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

FILTERS = [32, 64, 128, 256, 256, 64]

class RegressionSFCN(pl.LightningModule):
    """ Simple Fully Convolutional Network (SFCN) model in PyTorch Lightning.
    Adapted from https://doi.org/10.1016/j.media.2020.101871. A simple, VGG-like
    convolutional neural network for 3-dimensional neuroimaging data.
    """

    def __init__(self, *,
                 input_shape: Tuple[int] = (160, 1, 160, 192),  # (C, D, H, W)
                 pooling: Union[str, Callable] = 'avg',
                 include_top: bool = False,
                 dropout: float = 0.0,
                 activation: Union[str, Callable] = 'relu',
                 filters: List[int] = FILTERS,
                 weights: str = None,
                 name: str = 'sfcn',
                 prediction_range: Tuple[int] = (18,90),
                 restrict_strat: 'clamp',
                 seg_vector_len: int = 100,
                 **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.weights = weights
        self.name = name
        self.prediction_range = prediction_range
        self.restrict_strat = restrict_strat  
        # self.seg_fc = nn.Sequential(nn.Linear(64, 2))  

        self.include_top = include_top
        self.filters = filters
        self.activation = activation
        self.pooling = pooling
        self.dropout = dropout

        self.conv_blocks = self._build_conv_blocks(input_shape)

        # Bottleneck: 1x1x1 convolution followed by global pooling
        self.bottleneck_conv = Conv3d(self.filters[-2],self.filters[-1], kernel_size=(1,1,1), padding='same')
        self.bottleneck_norm = BatchNorm3d(self.filters[-1])
        self.bottleneck_activation = self._get_activation(activation)
        
        if self.pooling == 'avg':
            self.global_pool = AdaptiveAvgPool3d(1)
        elif self.pooling == 'max':
            self.global_pool = AdaptiveMaxPool3d(1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")

        # Prediction head (for include_top)
        if self.include_top:
            self.dropout_layer = Dropout(p=dropout)
            #self.prediction_head = self.prediction_head_seg_vol()
            self.prediction_head = self.prediction_head_fn() 

    def _build_conv_blocks(self, input_shape):
        layers = []
        in_channels = input_shape[0]

        for i, num_filters in enumerate(self.filters[:-1]):
            layers.append(Conv3d(in_channels, num_filters, kernel_size=(3, 3, 3), padding='same'))
            layers.append(BatchNorm3d(num_filters))
            layers.append(self._get_activation(self.activation))
            layers.append(MaxPool3d(kernel_size = (2,2,2), stride=2))
            in_channels = num_filters

        return nn.Sequential(*layers)

    def _get_activation(self, activation: Union[str, Callable]) -> Callable:
        if activation == 'relu':
            return nn.ReLU()
        # You can add more activation functions if needed (e.g., 'leaky_relu', 'tanh', etc.)
        raise ValueError(f"Unsupported activation: {activation}")

    def prediction_head_fn(self):
        lower = np.min(self.prediction_range)
        upper = np.max(self.prediction_range)
        layers = []
        layers.append(nn.Linear(64, 1))
        self.upper = upper 
        self.lower = lower
        return nn.Sequential(*layers)
    

    def forward(self, x, seg_vector=None):
        # Forward pass through convolution blocks
        x = x.to(torch.float32)
        x = self.conv_blocks(x)

        # Bottleneck
        x = self.bottleneck_conv(x)
        x = self.bottleneck_norm(x)
        x = self.bottleneck_activation(x)
  
        # Global pooling
        x = self.global_pool(x) 
    
        # Flatten the tensor
        x = torch.flatten(x,1)
        features = x

        if self.include_top:
            # Apply dropout (if include_top is True)
            x = self.dropout_layer(x)
            # Prediction head 
            if self.prediction_head:
                x = self.prediction_head(x)
            if self.restrict_strat == 'sigmoid':
                x = torch.sigmoid(5 * x) * (self.upper - self.lower) + self.lower
            elif self.restrict_strat == 'tan':
                x = 0.5 * (torch.tanh(x) + 1.0) * (self.upper - self.lower) + self.lower
            elif self.restrict_strat == 'clamp':
                x = torch.clamp(x, max=self.upper-self.lower)
                x = x + self.lower
        return x
    
    def training_step(self, batch, batch_idx):
        # Get the input data and target labels (ages) from the batch
        x, y = batch  

        # Forward pass: Get predictions
        y_pred, _ = self.forward(x)

        # Compute the loss: Mean Squared Error (MSE) for regression
        loss = F.mse_loss(y_pred.squeeze(), y.squeeze())  # Use .squeeze() to handle batch dimension

        # Log the training loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss
 
    def validation_step(self, batch, batch_idx):
        # Get the input data and target labels from the batch
        x, y = batch  

        # Forward pass: Get predictions
        y_pred, _ = self.forward(x)

        # Compute the loss: Mean Squared Error (MSE) for regression
        loss = F.mse_loss(y_pred.squeeze(), y.squeeze())

        # Log the validation loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)


 
