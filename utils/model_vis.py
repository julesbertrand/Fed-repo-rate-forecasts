import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from scipy.ndimage.interpolation import shift  # plot_model and compute_series_to_plot

from sklearn.metrics import r2_score
from sklearn.base import is_classifier, is_regressor

if not __name__ == "__main__":
    from utils.utils import *
    from utils.visualization import visualization_grid
    from utils.model_utils import regression_metrics, classification_metrics
else:
    from visualization import visualization_grid
    from model_utils import regression_metrics, classification_metrics


def plot_models(
    estimators: list,
    X_train,
    X_test,
    Y_train,
    Y_test,
    date_col: str,
    predicted_feature: str,
    plotted_feature: str = None,
    estimator_params: list = [],
    type_of_change=None,
    clip_to_zero: bool = False,
    is_regressor: bool = True,
    ncols: int = 1,
    height_per_ax: int = 6,
):
    if type_of_change not in [None, "diff", "pct_change"]:
        raise ValueError("type_of_change must be one of 'None', 'diff', 'pct_change'.")
    if not type_of_change:
        if plotted_feature != predicted_feature:
            plotted_feature = predicted_feature
            # warning to be implemented
    if not plotted_feature:
        plotted_feature = predicted_feature

    # handle subplot_params missing
    diff = len(estimators) - len(estimator_params)
    if diff > 0:
        for i in range(diff):
            estimator_params.append({})
    estimator_params = estimator_params[::-1]

    subplot_params = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "date_col": date_col,
        "plotted_feature": plotted_feature,
        "estimator_params": estimator_params,
        "type_of_change": type_of_change,
        "clip_to_zero": clip_to_zero,
        "is_regressor": is_regressor,
        "ncols": ncols,
    }

    @visualization_grid(give_grid_to_subplot_function=True)
    def subplot_model(
        grid_pos,
        estimator,
        text_font_size,
        X_train,
        Y_train,
        X_test,
        Y_test,
        date_col,
        plotted_feature,
        estimator_params,
        type_of_change,
        clip_to_zero,
        is_regressor,
        ncols,
    ):
        plot_df_train, plot_df_test = compute_series_to_plot(
            estimator=estimator,
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            date_col=date_col,
            plotted_feature=plotted_feature,
            type_of_change=type_of_change,
            clip_to_zero=clip_to_zero,
        )
        metrics = subplot_model_(
            grid_pos=grid_pos,
            estimator=estimator,
            plot_df_train=plot_df_train,
            plot_df_test=plot_df_test,
            date_col=date_col,
            plotted_feature=plotted_feature,
            plot_metrics=True,
            estimator_params=estimator_params.pop(),
            is_regressor=is_regressor,
            ncols=ncols,
        )
        return metrics

    fig_title = "Training and Test forecasting performance: '{:s}'".format(
        plotted_feature.replace("_", " ")
    )

    subplot_model(
        data=Y_train[["Date"]].append(Y_test[["Date"]]),
        date_col="Date",
        items=estimators,
        excl_items=[],
        ncols=1,
        fig_title=fig_title,
        height_per_ax=height_per_ax,
        subplot_params=subplot_params,
    )


def compute_series_to_plot(
    estimator,
    X_train,
    X_test,
    Y_train,
    Y_test,
    date_col,
    plotted_feature,
    type_of_change=None,
    clip_to_zero=False,
):
    pred_train = estimator.predict(X_train).reshape(-1)
    pred_test = estimator.predict(X_test).reshape(-1)
    plot_df_train = Y_train[[plotted_feature, date_col]]
    plot_df_test = Y_test[[plotted_feature, date_col]]
    if type_of_change is None:
        plot_df_train.loc[:, "pred_to_plot"] = pred_train
        if clip_to_zero:
            plot_df_test.loc[:, "pred_to_plot"] = pred_test.clip(min=0)
        else:
            plot_df_test.loc[:, "pred_to_plot"] = pred_test
        # if is_classifier(estimator):
        #     plot_df_test.loc[:, "Y_scores"] = list(estimator.predict_proba(X_test))
    else:  # compute reconstitued feature
        start = Y_train[plotted_feature].iloc[-1]
        if type_of_change == "diff":
            if clip_to_zero:
                plot_df_test.loc[:, "reconst"] = start
                for i, val in enumerate(pred_test):
                    new_val = val + plot_df_test["reconst"].iloc[i - 1]
                    if new_val < 0:
                        new_val = 0
                        # new_val = plot_df_test.loc[i - 1, "reconst"]
                    plot_df_test.loc[i, "reconst"] = new_val
            else:
                plot_df_test.loc[:, "reconst"] = start + pred_test.cumsum(axis=0)
            plot_df_train.loc[:, "pred_to_plot"] = Y_train[plotted_feature] + shift(
                pred_train, 1, cval=0
            )
            plot_df_test.loc[:, "pred_to_plot"] = Y_test[plotted_feature] + shift(
                pred_test, 1, cval=0
            )
        elif type_of_change == "pct_change":
            if clip_to_zero:
                plot_df_test.loc[:, "reconst"] = start
                for i, val in enumerate(pred_test):
                    new_val = val * plot_df_test["reconst"].iloc[i - 1]
                    if new_val < 0:
                        new_val = 0
                        # new_val = plot_df_test.loc[i - 1, "reconst"]
                    plot_df_test.loc[i, "reconst"] = new_val
            else:
                plot_df_test.loc[:, "reconst"] = start + pred_test.cumprod(axis=0)
            plot_df_train.loc[:, "pred_to_plot"] = Y_train[plotted_feature] * (
                1 + shift(pred_train, 1, cval=1)
            )
            plot_df_test.loc[:, "pred_to_plot"] = Y_test[plotted_feature] * (
                1 + shift(pred_test, 1, cval=1)
            )

    plot_df_train.loc[:, "abs_error"] = (
        plot_df_train["pred_to_plot"] - Y_train[plotted_feature]
    ).abs()
    plot_df_train.loc[:, "pct_error"] = (
        plot_df_train["abs_error"] / Y_train[plotted_feature].abs()
    )
    plot_df_test.loc[:, "abs_error"] = (
        plot_df_test["pred_to_plot"] - Y_test[plotted_feature]
    ).abs()
    plot_df_test.loc[:, "pct_error"] = (
        plot_df_test["abs_error"] / Y_test[plotted_feature].abs()
    )
    return plot_df_train, plot_df_test


def subplot_model_(
    grid_pos,
    estimator,
    plot_df_train,
    plot_df_test,
    date_col,
    plotted_feature,
    plot_metrics,
    estimator_params,
    is_regressor,
    ncols,
    text_font_size=10,
):
    # define grid position for the subplots
    fig, grid, top, bottom, horiz = grid_pos
    ax_main = fig.add_subplot(grid[top : bottom - 1, horiz])
    ax_bottom = fig.add_subplot(grid[bottom - 1 : bottom + 1, horiz])

    # plot estimator training and test values
    ax_main.plot(
        plot_df_train[date_col], plot_df_train[plotted_feature], label="Y_train"
    )
    ax_main.plot(
        plot_df_train[date_col],
        plot_df_train["pred_to_plot"],
        label="Training preds",
        color=sns.color_palette()[0],
        alpha=0.5,
    )
    ax_main.plot(plot_df_test[date_col], plot_df_test[plotted_feature], label="Y_test")
    ax_main.plot(
        plot_df_test[date_col], plot_df_test["pred_to_plot"], label="Test preds"
    )
    if "reconst" in plot_df_test.columns:
        # if reconstitution columns in df_test_plot, plot it
        ax_main.plot(
            plot_df_test[date_col],
            plot_df_test["reconst"],
            label="Reconstitution",
            color=sns.color_palette()[2],
            alpha=0.5,
        )
    ax_main.set_ylabel(plotted_feature.replace("_", " "), fontsize=text_font_size + 2)
    ax_main.legend(loc="best", ncol=1, fontsize=text_font_size)

    # abs error subplot below
    col_abs = sns.color_palette()[3]
    ax_bottom.plot(
        plot_df_train[date_col],
        plot_df_train["abs_error"],
        label="Abs error - training",
        color=col_abs,
    )
    ax_bottom.plot(
        plot_df_test[date_col],
        plot_df_test["abs_error"],
        label="Abs error - test",
        color=col_abs,
    )
    ax_bottom.set_ylabel("Abs error", fontsize=text_font_size + 2)
    handles_1, _ = ax_bottom.get_legend_handles_labels()

    # percentage error subplot below
    ax_bottom_2 = ax_bottom.twinx()
    col_pct = sns.color_palette()[4]
    ax_bottom_2.plot(
        plot_df_train[date_col],
        plot_df_train["pct_error"],
        label="Pct error - training",
        color=col_pct,
        alpha=0.8,
    )
    ax_bottom_2.plot(
        plot_df_test[date_col],
        plot_df_test["pct_error"],
        label="Pct error - test",
        color=col_pct,
        alpha=0.8,
    )
    ax_bottom_2.set_ylabel("% error (log)", fontsize=text_font_size + 2)
    ax_bottom_2.set_yscale("log")
    ax_bottom_2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_bottom_2.set_ylim(bottom=1e-2)
    ax_bottom_2.grid(linestyle="--", color="grey", alpha=0.8)
    handles_2, _ = ax_bottom_2.get_legend_handles_labels()
    ax_bottom_2.legend(
        [handles_1[0], handles_2[0]],
        ["Abs error", "% error"],
        ncol=1,
        fontsize=text_font_size,
        loc="best",
    )

    # vertical line separating training and test set
    x_line = plot_df_train[date_col].iloc[-1]
    ax_main.axvline(x=x_line, color="r", alpha=0.5, linestyle="--", linewidth=0.7)
    bottom, top = ax_main.get_ylim()
    y_txt = (top - bottom) * 0.95 + bottom
    ax_main.text(
        x=x_line,
        y=y_txt,
        s="  Test",
        va="center",
        ha="left",
        color="r",
        alpha=0.5,
        fontsize=text_font_size - 2,
    )
    ax_main.text(
        x=x_line,
        y=y_txt,
        s="Training  ",
        va="center",
        ha="right",
        color="r",
        alpha=0.5,
        fontsize=text_font_size - 2,
    )
    ax_bottom.axvline(x=x_line, color="r", alpha=0.5, linestyle="--", linewidth=0.7)

    # title of subplot containing estimator type, non-default params and reg/classif metrics
    if estimator.__class__.__name__ == "Pipeline":
        estimator_names = [e.__class__.__name__ for e in estimator]
        estimator_name = " + ".join(estimator_names)
    elif estimator.__class__.__name__ == "Model":
        estimator_name = estimator.name
    else:
        estimator_name = estimator.__class__.__name__

    title_str = "Model: " + estimator_name + "\n"
    if estimator_params:
        params_list = [k + ": {}".format(v) for k, v in estimator_params.items()]
        params_str = "Params: "
        length = len(params_str)
        for p in params_list:
            if length + len(p) > 150 and len(p) < 150:
                params_str = params_str[:-3]
                params_str += "\n"
                length = 0
            params_str += p + " | "
            length += len(p) + 3
        title_str += params_str
    title_str += "\n"

    # compute metrics
    if is_regressor:
        metrics_test = regression_metrics(
            plot_df_test[plotted_feature],
            plot_df_test["pred_to_plot"],
            return_series=False,
        )
    else:
        metrics_test = classification_metrics(
            plot_df_test[plotted_feature],
            plot_df_test["pred_to_plot"],
            plot_df_test["Y_scores"],
            return_series=False,
        )

    if plot_metrics:
        list_str = []
        if "reconst" in plot_df_test.columns:
            rr2 = r2_score(plot_df_test[plotted_feature], plot_df_test["reconst"])
            list_str += [r"Reconst $R^2$" + ": {:.4f}".format(rr2)]
        list_str += [
            "{:s}: {:.3f}".format(key, val) for key, val in metrics_test.items()
        ]
        metrics_str = []
        # handle metrics line width in subplot as a function of ncols
        idx = len(list_str) // ncols
        for i in range(ncols):
            metrics_str += ["  |  ".join(list_str[i * idx : (i + 1) * idx])]
        metrics_str = "\n".join(metrics_str)
        title_str += metrics_str
    
    ax_main.set_title(title_str, fontsize=text_font_size + 2)
    return metrics_test


def plot_feature_importance(fitted_model, features, threshold=0.01, pca=None):
    """
    Compute and plot feature importance for tree based methods from sklearn or similar
    input: model already fitted
        features: names of the features
        threshold: minimum feature importance for the feature to be plotted
        pca: if a pca was applied, giev the pca object to find back the real feature importance
    """
    # obj.feature_importances_ will raise AttributeError if the fitted_model does not have it
    importance = fitted_model.feature_importances_
    if pca:
        # importance is of size (n_components, 1)
        P = pca.components_  # shape (n_components, n_features)
        importance = np.dot(P.T, importance)
    idx = [x[0] for x in enumerate(importance) if x[1] > threshold]
    labels = features[idx]
    importance = importance[idx]
    idx = np.argsort(importance)[::-1]
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(8, max(8, 0.2 * len(idx))))
    sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
    plt.title("Feature importance for current model", fontsize=18)
    for i, val in enumerate(importance[idx]):
        plt.text(val + 0.01, i, s="{:.3f}".format(val), ha="left", va="center")
    plt.xticks(fontsize=14)
    plt.gca().set_xlim(0, max(importance[idx]) + 0.03)
    plt.show()
