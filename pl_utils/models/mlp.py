"""Multi Layer Perceptron Model."""

from collections import OrderedDict

import lightning as L

import torch


# pylint: disable=too-many-ancestors
class MLPModel(L.LightningModule):

    def __init__(self, layer_dims='1 100 1', lr=0.001):
        """Initialize fully connected layers."""
        super().__init__()

        self.layer_dims = layer_dims
        self.lr = lr

        layer_dims_list_str = self.layer_dims.split()
        layer_dims = [int(layer_dim)
                      for layer_dim in layer_dims_list_str]
        mlp_layers = OrderedDict()

        for layer_idx, _ in enumerate(layer_dims[0:-2]):
            mlp_layers['linear' + str(layer_idx + 1)] = \
                torch.nn.Linear(layer_dims[layer_idx],
                                layer_dims[layer_idx + 1])
            mlp_layers['silu' + str(layer_idx + 1)] = torch.nn.SiLU()

        layer_idx = len(layer_dims) - 2
        mlp_layers['linear' + str(layer_idx + 1)] = \
            torch.nn.Linear(layer_dims[layer_idx],
                            layer_dims[layer_idx + 1])

        self.model = torch.nn.Sequential(mlp_layers)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        return self.model(inputs)

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return super().configure_optimizers()
        # optimizer = torch.optim.Adam(self.parameters())
        # return optimizer

    # pylint: disable=unused-argument
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # pylint: disable=unused-argument
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # pylint: disable=unused-argument
    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        return outputs
