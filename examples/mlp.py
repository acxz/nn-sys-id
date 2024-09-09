"""Train a model from a training dataset."""

import pl_utils as plu

import lightning as L
from lightning.pytorch.cli import LightningCLI

import torch


class VectorDataset(torch.utils.data.Dataset):
    """Vector Dataset class to facilitate training."""

    def __init__(self, input_data, output_data):
        """Initialize input and output dataset."""
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        """Compute length of dataset."""
        return len(self.input_data)

    def __getitem__(self, idx):
        """Recover an item of dataset."""
        return self.input_data[idx], self.output_data[idx]


# pylint: disable=abstract-method
# pylint: disable=too-many-instance-attributes
class VectorDataModule(L.LightningDataModule):
    """Data module to load train/val/test dataloaders."""

    def __init__(self, num_workers=1, batch_size=1):
        """Initialze variables."""
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size


    def setup(self, stage=None):
        """Create and assign splits."""
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # Create data
        # TODO: is this best practice to create data inside of setup function?
        samples = 100000
        input_dim = 1
        output_dim = 3
        data = torch.empty(samples, input_dim + output_dim)

        time = torch.linspace(1, 5.7, samples)
        for sample_idx in range(samples):
            data[sample_idx] = torch.Tensor([time[sample_idx],
                                             torch.cos(2 * torch.pi *
                                                       time[sample_idx]),
                                             torch.sin(2 * torch.pi *
                                                       time[sample_idx]),
                                             time[sample_idx]])

        train_pct = 0.8
        test_pct = 0.1

        samples = len(data)
        train_samples = int(train_pct * samples)
        test_samples = int(test_pct * samples)
        val_samples = samples - train_samples - test_samples

        input_data, output_data = self._split_input_output_data(data)
        vector_dataset = VectorDataset(input_data, output_data)
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                vector_dataset, [train_samples, val_samples, test_samples])
        self.predict_dataset = vector_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size)

    @staticmethod
    def _split_input_output_data(data):
        input_dim = 1
        input_data = data[:, 0:input_dim]
        output_data = data[:, input_dim:None]
        return input_data, output_data


class ONNXExport(L.pytorch.callbacks.Callback):
    def on_train_end(self, trainer, pl_module):
        onnx_filepath = 'mlp.onnx'
        onnx_batch_size = 1
        input_dim = 1
        input_sample = torch.randn((onnx_batch_size, input_dim))
        pl_module.to_onnx(onnx_filepath, input_sample, export_params=True)
        print(onnx_filepath + " exported.")


def cli_main():
    cli = LightningCLI(model_class=plu.models.mlp.MLPModel, datamodule_class=VectorDataModule)


if __name__ == '__main__':
    cli_main()
