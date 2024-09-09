"""Gaussian Process Model."""

import gpytorch

import lightning as L

import torch


class BIMOEGP(gpytorch.models.ExactGP):
    """batch independent multioutput exact gp model."""

    def __init__(self, train_input_data, train_output_data, likelihood):
        """Initialize gp model with mean and covar."""
        super().__init__(train_input_data, train_output_data, likelihood)

        output_dim = train_output_data.size(dim=1)
        output_dim_torch = torch.Size([output_dim])

        self.mean_module = \
            gpytorch.means.ConstantMean(batch_shape=output_dim_torch)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=output_dim_torch),
            batch_shape=output_dim_torch)

        # Receiving error when using jit:
        # RuntimeError: mean shape torch.Size([9, 2]) is incompatible with covariance shape torch.Size([144, 144])
        # Something wrong with prediction stage

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Compute prediction."""
        mean = self.mean_module(input_)
        covar = self.covar_module(input_)

        return \
            gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean, covar))


# FIXME: may be used for torch jit tracing for model export
class MeanVarModelWrapper(torch.nn.Module):
    """Wrapper class to output prediction model."""

    def __init__(self, gp):
        """Initialize gp."""
        super().__init__()
        self.gp = gp

    def forward(self, input_):
        """Compute prediction."""
        output_dist = self.gp(input_)
        return output_dist.mean, output_dist.variance


# pylint: disable=too-many-ancestors
class BIMOEGPModel(L.LightningModule):
    """batch independent multioutput exact gp model."""

    def __init__(self, train_input_data, train_output_data):
        """Initialize gp model with mean and covar."""
        super().__init__()

        # take in train dataloader as the input tbh

        self.train_input_data = train_input_data
        self.train_output_data = train_output_data

        output_dim = self.train_output_data.shape[1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=output_dim)

        self.bimoegp = BIMOEGP(self.train_input_data,
                               self.train_output_data,
                               self.likelihood)

        # Why does the mll need the likelihood separately from the model (when the model already has a likelihood?)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.bimoegp)

    # pylint: disable=arguments-differ
    def forward(self, input_):
        """Compute prediction."""
        return self.bimoegp(input_)

    # pylint: disable=unused-argument
    def training_step(self, batch, batch_idx):
        input_, target = batch
        output = self(input_)
        loss = -self.mll(output, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        #return super().configure_optimizers()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.832)
        return optimizer

    # pylint: disable=unused-argument
    def validation_step(self, batch, batch_idx):
        input_, target = batch
        output = self(input_)
        loss = -self.mll(output, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # pylint: disable=unused-argument
    def test_step(self, batch, batch_idx):
        input_, target = batch
        output = self(input_)
        loss = -self.mll(output, target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # pylint: disable=unused-argument
    def predict_step(self, batch, batch_idx):
        input_, target = batch
        output = self(input_)
        return output
