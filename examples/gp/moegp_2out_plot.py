# input not shown, 2d plot with (x,y) and covariance ellipses (tubed)

import math

import pl_utils as plu

import torch

def main():
    # create some input data to be predicted on
    samples = 1000
    freq = 2 * math.pi
    x_domain = 1
    predict_input_data = torch.linspace(0, x_domain, samples).unsqueeze(1)

    # load model
    pt_path = 'moegp_1in2out.pt'
    checkpoint = torch.load(pt_path)
    model_state_dict = checkpoint['model_state_dict']
    train_input_data = checkpoint['train_input_data']
    train_output_data = checkpoint['train_output_data']
    model = plu.models.gp.MOEGPModel(train_input_data, train_output_data)
    model.load_state_dict(model_state_dict)

    # predict on mode
    model.eval()
    with torch.no_grad():
        predictions = model.likelihood(model(predict_input_data))
        mean = predictions.mean
        # TODO: logic for getting confidence ellipses
        lower, upper = predictions.confidence_region()
        import pdb; pdb.set_trace()

    # plots
    # pylint: disable=import-outside-toplevel
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=torch.cos(predict_input_data[:, 0] * freq),
            y=torch.sin(predict_input_data[:, 0] * freq),
            mode='lines',
            line={
                'color': 'blue',
                'dash': 'dot',
            },
            name='func',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=model.train_output_data[:, 0],
            y=model.train_output_data[:, 1],
            mode='markers',
            marker={
                'size': 10,
                'color': 'blue',
            },
            name='data',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mean[:, 0],
            y=mean[:, 1],
            mode='lines',
            line={
                'color': 'blue',
            },
            name='posterior mean',
            showlegend=True
        )
    )

    fig.show()


if __name__ == '__main__':
    main()
