###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "inline")
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def count(df, col_analysis):
    df = df[df["income"] == ">50K"]
    new_df = df[col_analysis].value_counts().reset_index()
    x = new_df["index"].astype(str).tolist()
    y = new_df[col_analysis].tolist()
    return x, y


def log_transformation(df, cols):
    df[cols] = df[cols].apply(lambda x: np.log(x + 1))
    return df[cols]


def compare_visually(data):
    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.15)
    fig.add_trace(
        go.Bar(name="Gender", x=count(data, "sex")[0], y=count(data, "sex")[1]),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Workclass",
            x=count(data, "workclass")[0],
            y=count(data, "workclass")[1],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(name="Race", x=count(data, "race")[0], y=count(data, "race")[1]),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Marital Status",
            x=count(data, "marital-status")[0],
            y=count(data, "marital-status")[1],
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Income Per Weekly Hours Rate",
            x=count(data, "hours-per-week")[0],
            y=count(data, "hours-per-week")[1],
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Relationship",
            x=count(data, "relationship")[0],
            y=count(data, "relationship")[1],
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Education Level",
            x=count(data, "education_level")[0],
            y=count(data, "education_level")[1],
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name="Occupation",
            x=count(data, "occupation")[0],
            y=count(data, "occupation")[1],
        ),
        row=4,
        col=2,
    )
    fig.update_layout(height=700, width=1000, title_text="Features per Income")
    return fig.show()


def distribution(df, col1, col2, transformed=False):
    dist = make_subplots(rows=1, cols=2, vertical_spacing=0.3)
    if transformed == False:
        dist.add_trace(
            go.Histogram(name="capital-gain", x=df[col1].tolist()), row=1, col=1
        )
        dist.add_trace(
            go.Histogram(name="capital-loss", x=df[col2].tolist()), row=1, col=2
        )
        dist.update_layout(
            title_text="Skewed Distributions of Continuous Census Data Features"
        )
    else:
        df[[col1, col2]] = df[[col1, col2]].apply(lambda x: np.log(x + 1))
        dist.add_trace(
            go.Histogram(name="capital-gain", x=df[col1].tolist()), row=1, col=1
        )
        dist.add_trace(
            go.Histogram(name="capital-loss", x=df[col2].tolist()), row=1, col=2
        )
        dist.update_layout(
            title_text="Log-transformed Distributions of Continuous Census Data Features"
        )
    dist.update_yaxes(
        range=[0, 2000],
        tickvals=[0, 500, 1000, 1500, 2000],
        ticktext=["0", "500", "1000", "1500", ">2000"],
    )
    return dist.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.

    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """

    # Create figure
    fig, ax = pl.subplots(2, 3, figsize=(11, 7))

    # Constants
    bar_width = 0.3
    colors = ["#A00000", "#00A0A0", "#00A000"]

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(
            ["train_time", "acc_train", "f_train", "pred_time", "acc_test", "f_test"]
        ):
            for i in np.arange(3):

                # Creative plot code
                ax[j // 3, j % 3].bar(
                    i + k * bar_width,
                    results[learner][i][metric],
                    width=bar_width,
                    color=colors[k],
                )
                ax[j // 3, j % 3].set_xticks([0.45, 1.45, 2.45])
                ax[j // 3, j % 3].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 3, j % 3].set_xlabel("Training Set Size")
                ax[j // 3, j % 3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 1].axhline(
        y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[0, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )
    ax[1, 2].axhline(
        y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color="k", linestyle="dashed"
    )

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    pl.legend(
        handles=patches,
        bbox_to_anchor=(-0.80, 2.53),
        loc="upper center",
        borderaxespad=0.0,
        ncol=3,
        fontsize="x-large",
    )

    # Aesthetics
    pl.suptitle(
        "Performance Metrics for Three Supervised Learning Models", fontsize=16, y=1.10
    )
    pl.tight_layout()
    pl.show()


def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize=(20, 5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    pl.bar(
        np.arange(5),
        values,
        width=0.6,
        align="center",
        color="#00A000",
        label="Feature Weight",
    )
    pl.bar(
        np.arange(5) - 0.3,
        np.cumsum(values),
        width=0.2,
        align="center",
        color="#00A0A0",
        label="Cumulative Feature Weight",
    )
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize=12)
    pl.xlabel("Feature", fontsize=12)

    pl.legend(loc="upper center")
    pl.tight_layout()
    pl.show()
