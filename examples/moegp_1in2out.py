"""Train a model from a training dataset."""

import gpytorch

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

        # TODO: is this needed?
        # flag to ensure setup (i.e. random split) only happens once
        self.has_setup = False


    def setup(self, stage=None):
        """Create and assign splits."""
        if self.has_setup:
            return
        self.has_setup = True

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        # TODO: is this best practice to create data inside of setup function?
        # Create data to fit
        # samples = 100000
        # input_dim = 1
        # output_dim = 3
        # data = torch.empty(samples, input_dim + output_dim)

        # time = torch.linspace(1, 5.7, samples)
        # for sample_idx in range(samples):
        #     data[sample_idx] = torch.Tensor([time[sample_idx],
        #                                      torch.cos(2 * torch.pi *
        #                                                time[sample_idx]),
        #                                      torch.sin(2 * torch.pi *
        #                                                time[sample_idx]),
        #                                      time[sample_idx]])

        # Create data to fit
        torch.manual_seed(3)
        samples = 100
        noise_scale = 0.05
        freq = 4 * torch.pi
        x_domain = 1
        input_data = torch.linspace(0, x_domain, samples).unsqueeze(1)
        sin_output_data = torch.sin(input_data * freq) + \
            torch.randn(input_data.size()) * noise_scale
        cos_output_data = torch.cos(input_data * freq) + \
            torch.randn(input_data.size()) * noise_scale
        data = torch.cat([input_data, sin_output_data, cos_output_data],
                                    1)
        random_indices = torch.randperm(data.shape[0])
        data = data[random_indices]
        data = data[0:10]

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
            # batch size must be the training dataset for GPs
            batch_size=len(self.train_dataset))

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


# pylint: disable=too-many-locals
def cli_main():
    #cli = LightningCLI(model_class=plu.models.gp.MOEGPModel, datamodule_class=VectorDataModule)

    # setup data module manually since it needs train_input/output_data during
    # initialization, which also need to be the same ones used in training
    print("where are we")
    data_module = VectorDataModule()
    print("seg fault?")
    data_module.setup()
    train_input_data, train_output_data = data_module.train_dataset[:]

    # create model
    model = plu.models.gp.MOEGPModel(train_input_data,
                                       train_output_data)

    # train model
    trainer = L.Trainer(max_epochs=50)
    trainer.fit(model=model, train_dataloaders=data_module.train_dataloader())

    # export model
    pt_filepath = 'moegp_1in2out.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'train_input_data': train_input_data,
                'train_output_data': train_output_data},
               pt_filepath)

"""
    training_args_list = ['--accelerator', 'auto',
                          '--accumulate_grad_batches', '1',
                          '--auto_lr_find', 'False',
                          '--benchmark', 'True',
                          '--enable_checkpointing', 'True',
                          '--detect_anomaly', 'True',
                          '--fast_dev_run', 'False',
                          '--enable_progress_bar', 'True',
                          '--max_epochs', '50',
                          '--enable_model_summary', 'True']

    model_args_list = ['--learning_rate', '0.832']

    # export model in onnx
    # below does not work yet
    # onnx_filepath = 'gp_example.onnx'
    # onnx_batch_size = 1
    # input_dim = 1
    # input_sample = torch.randn((onnx_batch_size, input_dim))

    # model.to_onnx(onnx_filepath, input_sample, export_params=True)

    # with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
    #        gpytorch.settings.trace_mode():
    #    model.eval()
    #    model(input_sample)  # Do precomputation

    #    traced_model = torch.jit.trace(
    #        plu.models.gp.MeanVarModelWrapper(model), input_sample)

    # traced_model = torch.jit.onnx(onnx_filepath, input_sample,
    #                          export_params=True)
"""

if __name__ == '__main__':
    cli_main()
