import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
Functions:
cross_validate: cv based on sklearn grid searcgh enhanced for this problem
pca: perform pca and plot results
vif_analysis: perform vif_analysis
regression_metrics: compute regression metrics: r2, MAE, MAPE, RMSE,, MASE 
classification_metrics: compute classification metrics: accuracy, precision, recall, f1-score, ROC AUc, log loss
prepare_sequence_data: put data in sequences of shape (n_samples, lookback, n_features) for deep learning models
train_test_split: enhanced train test split to account for the dates here, using the scikit-learn one 
"""


def prepare_sequential_data(data, features, labels, lookback=12):
    if isinstance(data, pd.DataFrame):
        pass
    X_temp = data[features].to_numpy()
    Y = data[labels].iloc[:-lookback]
    n, p = X_temp.shape
    l = len(labels)
    X = np.zeros(shape=(n - lookback, lookback, p))
    for i in range(lookback):
        X[:, i, :] = X_temp[i + 1 : n - lookback + i + 1]
    return X, Y


def train_test_split_dates(
    X, Y, test_size=0.2, test_date=None, standardize=True, sequential=False
):
    if len(X) != len(Y):
        raise IndexError("X, Y, and dates must have the same number of rows")
    if not test_date:
        test_date_idx = int(len(X) * (1 - test_size))
        test_date = Y["Date"].iloc[test_date_idx]
    test_mask = (Y["Date"] >= test_date).to_numpy()
    X_train = X[~test_mask]
    X_test = X[test_mask]
    Y_train = Y[~test_mask]
    Y_test = Y[test_mask]
    if standardize:
        scaler = StandardScaler()
        if sequential:
            # sequential data of size (n_samples, lookback, n_features)
            scaler.fit(X_train[:, 0, :])
            for i in range(X.shape[1]):
                X_train[:, i, :] = scaler.transform(X_train[:, i, :])
                X_test[:, i, :] = scaler.transform(X_test[:, i, :])
        else:  # non-sequential data of size (n_samples, n_features)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
    return X_train, Y_train, X_test, Y_test


def cross_validate(
    X_train, Y_train, estimator, param_grid={}, n_splits=5, scoring=metrics.r2_score
):
    cv_params = {}
    fixed_params = {}
    n_permut = 1
    for key, val in param_grid.items():
        if len(val) == 1:
            fixed_params[key] = val[0]
        else:
            cv_params[key] = val
            n_permut *= len(val)
    print("\n" + " Feature to be predicted: ".center(120, "-"))
    print(Y_train.name)
    print("\n" + " Estimator: ".center(120, "-"))
    print(estimator.__class__.__name__)
    print("\n" + " Metric for evaluation: ".center(120, "-"))
    print(scoring if isinstance(scoring, str) else scoring.__name__)
    if len(fixed_params.keys()) > 0:
        print("\n" + " Fixed params: ".center(120, "-"))
        [print(key, value) for key, value in fixed_params.items()]
    print("\n" + " Params to be cross-validated: ".center(120, "-"))
    [print(key, value) for key, value in cv_params.items()]
    print(
        "\n # of permutations to be cross-validated: {:d} \n # of works: {:d}".format(
            n_permut, n_permut * n_splits
        )
    )
    answer = input("\n" + "Continue with these c-v parameters ? (y/n)  ")
    if answer == "n" or answer == "no":
        print("Please redefine inputs.")
        return

    splitter = TimeSeriesSplit(n_splits=n_splits)
    g = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=splitter,
        refit=True,
        verbose=1,
        n_jobs=-1,
        error_score="raise",
    )
    g.fit(X_train, Y_train)
    print("\n" + " Cross_validation finished ".center(120, "-"))
    return g


def pca(data, columns=[], excl_cols=[], n_components=10, plot_explained_var=True):
    if not columns:
        columns = data.columns
    columns = [
        c
        for c in columns
        if c in data.columns
        and c not in excl_cols
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    pca_ = PCA(n_components=n_components)
    pca_.fit(data[columns])
    if plot_explained_var:
        plt.style.use("seaborn-darkgrid")
        expl_variance = [0]
        for i in range(n_components):
            expl_variance.append(expl_variance[i] + pca_.explained_variance_ratio_[i])
        plt.plot(range(n_components + 1), expl_variance)
    return pca_


def vif_analysis(data, columns=[], excl_cols=[], threshold=10):
    if not columns:
        columns = data.columns
    columns = [
        c
        for c in columns
        if c in data.columns
        and c not in excl_cols
        and pd.api.types.is_numeric_dtype(data[c])
    ]

    excl_columns_vif = {}
    data_vif = data[columns]
    data_vif = data_vif.dropna()
    n_cols = len(data_vif.columns)
    print(" VIF analysis starting: {:d} features ".format(n_cols).center(120, "-"))

    while True:
        if n_cols <= 1:
            print(" VIF analysis stopped: only one feature remaining ".center(120, "-"))
            vif_final = pd.DataFrame({"features": data_vif.columns, "vif": [None]})
            break
        vif_values = [
            variance_inflation_factor(data_vif.values, i)
            for i in range(data_vif.shape[1])
        ]
        max_vif_idx = np.argmax(vif_values)
        max_vif_value = vif_values[max_vif_idx]
        if max_vif_value <= threshold:
            vif_final = pd.DataFrame({"features": data_vif.columns, "vif": vif_values})
            break
        max_vif_feature = data_vif.columns[max_vif_idx]
        excl_columns_vif[max_vif_feature] = max_vif_value
        data_vif.drop(columns=[max_vif_feature], inplace=True)
        n_cols -= 1

    print(" VIF analysis succesfully completed ".center(120, "-"))
    print("remaining features: {:d}".format(n_cols))
    print("excluded features: {:d}".format(len(excl_columns_vif)))
    return vif_final, excl_columns_vif


def regression_metrics(
    Y_test,
    Y_pred,
    return_series=False,
):
    """
    Compute several different regression metrics (MAE, RMSE, MAPE, ...) from Y_test and predictions
    """
    diff = Y_test - Y_pred
    local_metrics = {
        r"Test $R^2$": metrics.r2_score(Y_test, Y_pred),
        "ME": np.mean(diff),
        "MAE": np.mean(np.abs(diff)),
        "RMSE": np.sqrt(np.mean(diff ** 2)),
        "MPE": np.mean(diff / np.mean(Y_test)),
        # mean absolute percetage error
        "MAPE": np.mean(np.abs(np.where(Y_test != 0, diff / Y_test, 0))),
        # mean absolute scaled error
        "MASE": np.mean(np.abs(diff / np.mean(np.abs(np.diff(Y_test))))),
    }
    if return_series:
        series = {"Abs error": diff, "% error": diff / Y_test}
        return local_metrics, series
    return local_metrics


def classification_metrics(
    Y_test,
    Y_pred,
    Y_scores,
    labels=None,
    return_series=False,
):
    """
    Compute several different clasification metrics (accuracy, F1 score, AUC, ...) from Y_test and predictions
    """
    Y_scores = Y_scores.to_list()
    if labels is None:
        n_class = Y_scores[0].shape[0]
        labels = np.arange(-(n_class // 2), n_class // 2 + 1, 1.0)
    # ovo and labels needed as Y_test does not always contain all classes
    local_metrics = {
        "Acc": metrics.accuracy_score(Y_test, Y_pred),
        "Precision": metrics.precision_score(Y_test, Y_pred, average="weighted"),
        "Recall": metrics.recall_score(Y_test, Y_pred, average="weighted"),
        "F1": metrics.f1_score(Y_test, Y_pred, average="weighted"),
        "ROCAUC": metrics.roc_auc_score(
            Y_test, Y_scores, multi_class="ovo", average="weighted", labels=labels
        ),
        "LOGLOSS": metrics.log_loss(Y_test, Y_scores, labels=labels),
    }
    if return_series:
        series = {"Abs error": diff, "% error": diff / Y_test}
        return local_metrics, series
    return local_metrics
