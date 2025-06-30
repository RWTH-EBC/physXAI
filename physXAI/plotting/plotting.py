import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from physXAI.preprocessing.training_data import TrainingDataMultiStep, TrainingDataGeneric


def subplots(header: str, *figures) -> go.Figure:
    subplot_titles = []
    plot_type = []

    # Define titles and specs
    for fig_dict in figures:
        subplot_titles.append(fig_dict["title"])

        plot_type.append([{"type": fig_dict["type"]}])

    # Create the subplots
    fig = make_subplots(rows=len(figures), cols=1, subplot_titles=subplot_titles, specs=plot_type)

    # Add each figure's traces to the subplots
    for i, fig_dict in enumerate(figures):
        figure = fig_dict["figure"]

        # For specifically table type figures
        for trace in figure["data"]:
            fig.add_trace(trace, row=i + 1, col=1)
        if fig_dict["type"] == "scatter":
            fig.update_xaxes(title_text=figure.layout.xaxis.title.text, row=i + 1, col=1)
            fig.update_yaxes(title_text=figure.layout.yaxis.title.text, row=i + 1, col=1)

    fig.update_layout(title_text=header, height=300*len(figures), width=1250)

    fig.show()

    return fig


def plot_metrics_table(td: TrainingDataGeneric) -> go.Figure:
    # Extracting metrics and corresponding values
    metrics = td.metrics
    labels, values = metrics.get_metrics()

    # Creating the table
    fig = go.Figure()
    fig.add_trace(go.Table(
            header=dict(values=['Metrics', 'Value']),
            cells=dict(values=[labels, values])
        ))

    return fig


def plot_prediction_correlation(td: TrainingDataGeneric) -> go.Figure:
    fig1 = go.Figure()

    x = np.linspace(np.min(td.y_train_single), np.max(td.y_train_single), 100)
    fig1.add_trace(go.Scatter(x=x, y=x, mode='lines', line=dict(color='black', dash='dash'), name='Ref'))

    fig1.add_trace(go.Scatter(x=td.y_train_single[:, 0], y=td.y_train_pred_single[:, 0], name='Train', mode='markers',
                              marker=dict(color='green')))
    if td.y_val_single is not None:
        fig1.add_trace(
            go.Scatter(x=td.y_val_single[:, 0], y=td.y_val_pred_single[:, 0], name='Val', mode='markers',
                       marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=td.y_test_single[:, 0], y=td.y_test_pred_single[:, 0], name='Test', mode='markers',
                              marker=dict(color='red')))

    fig1.update_layout(
        xaxis_title="True Values",
        yaxis_title="Predicted Values"
    )
    return fig1


def plot_predictions(td: TrainingDataGeneric) -> go.Figure:

    y_train = td.y_train_single[:, 0]
    y_train_pred = td.y_train_pred_single
    if td.y_val_single is not None:
        y_val = td.y_val_single[:, 0]
        y_val_pred = td.y_val_pred_single
    else:
        y_val = np.array([])
        y_val_pred = np.array([])
    y_test = td.y_test_single[:, 0]
    y_test_pred = td.y_test_pred_single

    # Plot the predictions using plotly library
    y_train_combined = list(zip(y_train, y_train_pred, ['Train'] * len(y_train)))
    y_val_combined = list(zip(y_val, y_val_pred, ['Val'] * len(y_val)))
    y_test_combined = list(zip(y_test, y_test_pred, ['Test'] * len(y_test)))
    combined = y_train_combined + y_val_combined + y_test_combined
    combined.sort(key=lambda x: x[0])
    y_sorted, y_pred_sorted, labels = zip(*combined)

    df_plotly = pd.DataFrame(combined, columns=['y_sorted', 'y_pred_sorted', 'labels'])
    df_plotly['y_pred_sorted'] = df_plotly['y_pred_sorted'].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(len(y_sorted))), y=df_plotly['y_sorted'],
                              mode='markers',
                              name='True values',
                              marker=dict(color='black')))

    color_map = {
        'Train': 'green',
        'Val': 'blue',
        'Test': 'red',
    }

    fig2 = px.scatter(df_plotly, x=list(range(len(y_sorted))), y='y_pred_sorted', color='labels',
                      color_discrete_map=color_map)

    fig = go.Figure(data=fig1.data + fig2.data)

    fig.update_layout(
        xaxis_title="Sample",
        yaxis_title="Value"
    )

    return fig


def plot_training_history(td: TrainingDataGeneric) -> go.Figure:
    history = td.training_record.history
    # Plot training history using plotly library
    training_rmse = history['rmse']
    len_train = len(training_rmse)
    if td.y_val is not None:
        validation_rmse = history['val_rmse']
        len_val = len(validation_rmse)
    else:
        validation_rmse = []
        len_val = 0

    data = {
        'Epochs': list(range(1, len_train + 1)) + list(range(1, len_val + 1)),
        'RMSE': training_rmse + validation_rmse,
        'Type': ['Training RMSE'] * len_train + ['Validation RMSE'] * len_val
    }

    df = pd.DataFrame(data)

    color_map = {
        'Training RMSE': 'green',
        'Validation RMSE': 'blue',
    }

    fig = px.line(df, x='Epochs', y='RMSE', color='Type', markers=True, color_discrete_map=color_map)

    return fig


def plot_multi_rmse(td: TrainingDataMultiStep) -> go.Figure:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=np.array(range(1, len(td.metrics.rmse_train_l)+1)), y=np.array(td.metrics.rmse_train_l),
                              name='Train', mode='lines', marker=dict(color='green')))
    if td.y_val is not None:
        fig1.add_trace(go.Scatter(x=np.array(range(1, len(td.metrics.rmse_val_l) + 1)),
                                  y=np.array(td.metrics.rmse_val_l), name='Val',
                                  mode='lines', marker=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=np.array(range(1, len(td.metrics.rmse_test_l)+1)), y=np.array(td.metrics.rmse_test_l),
                              name='Test', mode='lines', marker=dict(color='red')))

    fig1.update_layout(
        xaxis_title="Prediction Step",
        yaxis_title="RMSE"
    )

    return fig1


def plot_recFeatureSelection(fs: dict, multi_step: bool, use_multi_step_error: bool):

    x = list()
    y = list()
    y_single = list()

    for k, v in fs.items():

        x.append(k)
        values = list()
        values_single = list()
        for f in v:
            values.append(f['kpi'])
            if multi_step:
                values_single.append(f['kpi_single_step'])

        if not multi_step:
            y.append(min(values))
        elif use_multi_step_error:
            index = values.index(min(values))
            y.append(values[index])
            y_single.append(values_single[index])
        else:
            index = values_single.index(min(values_single))
            y.append(values[index])
            y_single.append(values_single[index])

    fig = go.Figure()
    if not multi_step:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Single Step',
                                 marker=dict(color='blue')))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Multi Step',
                                 marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=x, y=y_single, mode='lines+markers', name='Single Step',
                                 marker=dict(color='green')))

    fig.update_layout(
        xaxis_title="Number Features",
        yaxis_title="RMSE"
    )

    fig.show()
