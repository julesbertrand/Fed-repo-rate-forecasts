import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn import metrics
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    print("\n" + " Params to be tested: ".center(120, "-"))
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
