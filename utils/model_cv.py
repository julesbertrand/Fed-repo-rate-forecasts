import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm.notebook import tqdm

import itertools  # time_series_split_cv
import ast  # for litteral_eval in time_series_split_cv

from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import is_classifier, is_regressor

if not __name__ == "__main__":
    from utils.model_visualization import * 

SEED = 10

def cv_model(X_train,
             Y_train,
             X_test,
             Y_test,
             predicted_feature,
             estimator,
             cv_params=None,
             n_splits_cv=5,
             metric=metrics.r2_score,
             plot_feature_importance=True,
             plot_cv_scores=True,
             plot_model_perf=False,
             plot_reconstitution=False,
             reconstitution_feature=None,
             reconstitution_type_of_diff=None  # None or 'diff' or 'pct'                
            ):
    mean_results_cv = time_series_split_cv(X_train,
                                           Y_train[predicted_feature],
                                           estimator = estimator,
                                           n_splits = n_splits_cv,
                                           cv_params = cv_params,
                                           metric = metric
                                          )
    best_model = mean_results_cv.loc[mean_results_cv['validation score'].idxmax]
    with pd.option_context('max_colwidth', None):
        print("\n" + "Best model found through cross-validation: ".center(120, "-"))
        print(best_model)
    model = estimator.set_params(**best_model.loc['params'])
    model = model.fit(X_train, Y_train[predicted_feature])
    
    if plot_feature_importance:
        feature_importance(model, X_train.columns)
    if plot_model_perf:
        _, _ = plot_model(X_train, 
                           Y_train,
                           X_test,
                           Y_test,
                           date_col='Date',
                           predicted_feature=predicted_feature,
                           models=[model],
                           model_params=[best_model.loc['params']],
                           ncols=1
                          )
    if plot_reconstitution:
        _, _ = plot_model(X_train, 
                           Y_train,
                           X_test,
                           Y_test,
                           date_col='Date',
                           predicted_feature=predicted_feature,
                           models=[model],
                           model_params=[best_model.loc['params']],
                           type_of_change=reconstitution_type_of_diff,
                           plotted_feature=reconstitution_feature,
                           ncols=1
                          )
    return mean_results_cv, model

def time_series_split_cv(X_train,  # df or series
                         Y_train,  # series
                         estimator,
                         n_splits=5,
                         cv_params={},  # same format as in sklearn GridSearchCV
                         metric = metrics.r2_score
                        ):
    """
    Cross-validation from timeseries for all estimators from sklearn
    Input: X_train, Y_train, estimator from sklearn, n_splits=k for k-folds cv, params grid to test (GridSearch only)
    Output: Dataframe with columns params (dict type), # fold, training score and validation score 
    """
    print("\n" + " Feature to be predicted: ".center(120, "-"))
    print(Y_train.name)
    print("\n" + " Estimator: ".center(120, "-"))
    print(estimator.__class__.__name__)
    print("\n" + " Metric for evaluation: ".center(120, "-"))
    print(metric.__name__)
    print("\n" + " Params to be tested: ".center(120, "-"))
    [print(key, value) for key, value in cv_params.items()]
    params_list = list(itertools.product(*cv_params.values()))
    n_combi = len(params_list)
    print("\n" + " # of possible combinations to be cross-validated: {:d}".format(n_combi))
    answer = input("\n" + "Continue with these c-v parameters ? (y/n)  ")
    if answer == "n" or answer == "no":
        print("Please redefine inputs.")
        return
    i = 1
    scores = []
    iterator = TimeSeriesSplit(n_splits=n_splits).split(X_train)
    params_list = [dict(zip(cv_params.keys(), e)) for e in params_list]
    n_candidates = len(params_list)
    print("\n%s" % " Cross-validation starting for {:s}".format(
        estimator.__class__.__name__
        ).center(120, "-"))
    print(" Fitting {:d} folds for each of {:d} candidates, totalling {:d} fits \n".format(n_splits,
                                                                                           n_candidates,
                                                                                           n_splits * n_candidates
                                                                                          ))
    for tr_index, val_index in iterator:
        print(" Folder #{:d} starting ".format(i).center(120, "-"))
        X_tr, X_val = X_train.iloc[tr_index], X_train.iloc[val_index]
        y_tr, y_val = Y_train.iloc[tr_index], Y_train.iloc[val_index]
        for params in tqdm(params_list, total=n_candidates):
            model_cv = estimator.set_params(**params)
            model_cv.fit(X_tr, y_tr)
            y_pred_tr = model_cv.predict(X_tr)
            y_pred_val = model_cv.predict(X_val)
            scores.append([params,
                          i,
                          metric(y_true=y_tr, y_pred=y_pred_tr),
                          metric(y_true=y_val, y_pred=y_pred_val)
                         ])
        i += 1
    # Convert 'params' to str to perform pivot (otherwise error)
    results_cv = pd.DataFrame(scores,
                              columns=['params', 'fold', 'training score' ,'validation score']
                             )
    results_cv['params'] = results_cv['params'].apply(lambda x: str(x))
    
    # Pivot table to get the mean values for each params over all folds
    mean_results_cv = pd.pivot_table(results_cv, values=['training score', 'validation score'],
                                     index='params', aggfunc=np.mean
                                    )
    # Convert 'params' back to dict
    mean_results_cv.reset_index(level='params', inplace=True)
    mean_results_cv['params'] = mean_results_cv['params'].apply(lambda x: ast.literal_eval(x)) 
    mean_results_cv.sort_values('validation score', ascending=False, inplace=True)
    return mean_results_cv