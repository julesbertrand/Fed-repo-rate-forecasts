import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm.notebook import tqdm
# from decimal import Decimal
import itertools  # time_series_split_cv
from copy import deepcopy  # plot model to copy model_params list
from scipy.ndimage.interpolation import shift  # plot_model and compute_series_to_plot
import ast  # for litteral_eval in time_series_split_cv

from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import is_classifier, is_regressor

if not __name__ == "__main__":  # for test purposes
    from utils.visualization import visualization_basis

SEED = 10

def cv_model(X_train,
                Y_train,
                X_test,
                Y_test,
                predicted_feature,
                estimator,
                params_to_cv=None,
                n_splits_cv=5,
                plot_feature_importance=True,
                plot_model_perf=True,
                plot_reconstitution=False,
                reconstitution_feature=None,
                reconstitution_type_of_diff=None  # None or 'diff' or 'pct'                
               ):
    print("\n" + " Feature to be predicted: ".center(120, "-"))
    print(predicted_feature)
    print("\n" + " Estimator: ".center(120, "-"))
    print(estimator.__class__.__name__)
    if isinstance(params_to_cv, dict):
        cv_params = params_to_cv
    else:
        cv_params = define_cv_params(estimator, params_to_cv)
    
    answer = input("\n" + "Continue with this cv-params ? (y/n)  ")
    if answer == "n" or answer == "no":
        print("Please redefined inputs.")
        return
    mean_results_cv = time_series_split_cv(X_train,
                                           Y_train[predicted_feature],
                                           estimator = estimator,
                                           n_splits = n_splits_cv,
                                           cv_params = cv_params
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
        pred_metrics, pred_df = plot_model(X_train, 
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
        recons_metrics, recons_df = plot_model(X_train, 
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
        return {
            "results_cv": mean_results_cv,
            "best_model_fitted": model,
            "Predicted feature metrics": pred_metrics,
            "Predicted feature df": pred_df,
            "Reconstitution feature metrics": recons_metrics,
            "Reconstitution feature df": recons_df
        }
    return {
        "results_cv": mean_results_cv,
        "best_model_fitted": model,
        "Predicted feature metrics": pred_metrics,
        "Predicted feature df": pred_df,
    }

def define_cv_params(estimator, params_to_cv=None):
    name = estimator.__class__.__name__
    if name == "LinearDiscriminantAnalysis":
        cv_params = {}
    elif name == "LinearRegression":
        cv_params = {}
    elif name == "LogisticRegression":
        cv_params == {}
    elif name == "RandomForestRegressor" or name == "RandomForestClassifier":
        cv_params = {
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],  # Number of trees
            'max_features': [1, 'auto', 'sqrt', 'log2'],  # Number of features to consider at every split
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 6)] + [None],  # maximum depth of a tree
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
        }
    elif name == "GradientBoostingRegressor" or name == "GradientBoostingClassifier":
        cv_params = {
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 110, num = 6)],  # Number of trees
            'learning_rate': np.logspace(-3, -1, 3),  # weight of each tree in final estimator
            'max_depth': [3, 5, 7],  # maximum depth of a tree
            'min_samples_split': np.arange(2, int(np.sqrt(len(X_train.index))), 5),  # min number of obs in node to be considered for a split
            'max_features': ['sqrt', 'auto'],  # Number of features to consider at every split
            'min_samples_leaf': [1, 2, 3, 4],  # Minimum number of samples required at each leaf node
            'subsample': np.linspace(start=0.6, stop=1, num=5),
            'ccp_alpha': [0, 1e-4, 1e-2],  # Complexity parameter for tree prunning
        }
    elif name == "XGBRegressor" or name == "XGBClassifier":
        cv_params = {
            'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],  # Number of trees
            'eta': np.logspace(-2, -0.5, 3),  # weight of each tree in final estimator
            'max_depth': [3, 6, 10],  # maximum depth of a tree
            'min_child_weight': [1, 3, 6, 10],
            'subsample': np.linspace(start=0.6, stop=1, num=5),
#             'colsample_bytree': [0.3, 0.3, 0.7, 1],
            'gamma': [0, 1, 5, 10],
        }
    else:
        print("\n" + " No cross-validation params defined for this estimator yet ".center(120, "-"))
        cv_params = {}
    if params_to_cv is not None:
        temp = {}
        for param in cv_params.keys():
            if param in params_to_cv:
                temp[param] = cv_params[param]
        cv_params = temp
    print("\n" + " Params to be tested: ".center(120, "-"))
    [print(key, value) for key, value in cv_params.items()]
    n_combi = len(list(itertools.product(*cv_params.values())))
    print("\n" + " # of possible combinations to be cross-validated: {:d}".format(n_combi))
    return cv_params


def time_series_split_cv(X_train,
                         Y_train,
                         estimator,
                         n_splits=5,
                         cv_params={}  # same format as in sklearn GridSearchCV
                        ):
    """
    Cross-validation from timeseries for all estimators from sklearn
    Input: X_train, Y_train, estimator from sklearn, n_splits=k for k-folds cv, params grid to test (GridSearch only)
    Output: Dataframe with columns params (dict type), # fold, training score and validation score 
    """
    i = 1
    scores = []
    iterator = TimeSeriesSplit(n_splits=n_splits).split(X_train)
    params_list = list(itertools.product(*cv_params.values()))
    params_list = [dict(zip(cv_params.keys(), e)) for e in params_list]
    n_candidates = len(params_list)
    print("\n%s" % " {:d}-folds Cross-validation starting for {:s}".format(
        n_splits,
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
            model_cv = estimator.set_params(**params, n_jobs=-1)
            model_cv.fit(X_tr, y_tr)
            scores.append([params,
                          i,
                          model_cv.score(X_tr, y_tr),
                          model_cv.score(X_val, y_val)
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

def feature_importance(fitted_model, features, threshold=0.01):
    importance = fitted_model.feature_importances_
    idx = [x[0] for x in enumerate(importance) if x[1] > threshold]
    labels = features[idx]
    importance = importance[idx]
    idx = np.argsort(importance)[::-1]
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize = (8, max(8, 0.2 * len(idx))))
    sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
    plt.title("Feature importance for current model", fontsize=18)
    for i, val in enumerate(importance[idx]):
        plt.text(val + 0.01, i, s="{:.3f}".format(val), ha='left', va='center')
    plt.xticks(fontsize=14)
    plt.gca().set_xlim(0, max(importance[idx]) + 0.03)
    plt.show()

def plot_model(X_train, 
                Y_train,
                X_test,
                Y_test,
                date_col,
                predicted_feature,
                models,  # fitted model
                model_params=None,
                type_of_change=None,  # If None, plot predicted_feature. Else can be diff or pct_change
                plotted_feature=None,  # only if type_of_change is not none
                nb_periods=1,  # not defined yet, maybe the amount of periods to forecast forward
                plot_metrics=True,
                ncols=1
                ):
    """
    Give training and testing score, plot training and test set + training and test predictions
    Input: X_train, X_test, Y_train, Y_test 
        models: list of already fitted estimators
        predicted_feature_name: column of Y to be predicted
        date_col: column name of dates
        type_of change, plotted_feature: if the plotted feature is computed from the predictions, how to compute it (+ x or * (1 + x))
        nb_periods: predictions how many periods forward
        plot_metrics: plot_metrics in title of subplot if True
        model_params: params to be plotted in subplot title
        ncols: nb of columns in the figure
    Output: list of metrics for each models, list of plotted df for each model
    """
    assert_print = "Please provide predictions for either absolute difference or percentage of change per period"
    assert type_of_change in [None, 'diff', 'pct_change'], assert_print
    if type_of_change is None:
        plotted_feature = predicted_feature
    else:
        assert plotted_feature is not None, "Please provide a feature to plot"
    if model_params is None:
        model_params = [{}] * len(models)
    else:
        model_params = deepcopy(model_params)
    ncols = min(ncols, len(models))
    subplot_params = {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'date_col': date_col,
        'plotted_feature': plotted_feature,
        'type_of_change': type_of_change,
        'plot_metrics': plot_metrics,
        'model_params': model_params,
        'ncols': ncols,
        'nb_periods': nb_periods,
    }
    metrics = []
    df = []
    def compute_and_subplot_model(grid_position, model, text_font_size, **subplot_params):
        plot_df_train, plot_df_test = compute_series_to_plot(X_train=X_train,
                                                             Y_train=Y_train,
                                                             X_test=X_test,
                                                             Y_test=Y_test,
                                                             date_col=date_col,
                                                             plotted_feature=plotted_feature,
                                                             model=model,
                                                             type_of_change=type_of_change
                                                            )
        model_metrics = subplot_model(grid_position, 
                                      plot_df_train=plot_df_train,
                                      plot_df_test=plot_df_test,
                                      date_col=date_col,
                                      plotted_feature=plotted_feature,
                                      plot_metrics=plot_metrics, 
                                      model=model, 
                                      model_params=model_params[0],
                                      ncols=ncols,
                                      text_font_size=text_font_size
                                     )
        del model_params[0]
        metrics.append(model_metrics)
        df.append((plot_df_train, plot_df_test))
    fig_title = "Training and Test performance forecasting '{:s}' at {:d} month(s)".format(
        plotted_feature.replace("_", " "),
        nb_periods    
    )
    visualization_basis(data=Y_train[[date_col]].append(Y_test[[date_col]]),
                        subplot_function=compute_and_subplot_model,
                        subplot_params=subplot_params,
                        date_col='Date',
                        items=models,
                        excl_items=[],
                        ncols=ncols,
                        height_per_ax=6,  # must be an integer
                        # width_per_ax=5,  # must be an integer
                        fig_title=fig_title,
                        give_grid_to_subplot_function=True
                        )
    return metrics, df

def compute_series_to_plot(X_train,
                           X_test, 
                           Y_train, 
                           Y_test, 
                           date_col, 
                           plotted_feature, 
                           model, 
                           type_of_change=None,
                           nb_periods=1,
                          ):
        """
        Compute predictions dataframes to input subplot
        Input: same as plot_model
        Ouput: train and test df with Y_true, Y_pred, abs_error, pct_error
        """
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        plot_df_train = Y_train[[plotted_feature, date_col]]
        plot_df_test = Y_test[[plotted_feature, date_col]]
        if type_of_change is None:
            plot_df_train['pred_to_plot'] = pred_train
            plot_df_test['pred_to_plot'] = pred_test
            if is_classifier(model):
                plot_df_test['Y_scores'] = list(model.predict_proba(X_test))
        else:
            if type_of_change == 'diff':
                reconstitution = [Y_train[plotted_feature].iloc[-1]]
                for e in pred_test:
                    temps_val = reconstitution[-1] + e
                    reconstitution.append(temps_val if temps_val >= 0 else 0)
                plot_df_test['reconstitution'] = reconstitution[1:]
                # plot_df_test['reconstitution'] = Y_train[plotted_feature].iloc[-1] + pred_test.cumsum(axis=0)
                plot_df_train['pred_to_plot'] = Y_train[plotted_feature] + shift(pred_train, 1, cval=0)
                plot_df_test['pred_to_plot'] = Y_test[plotted_feature] + shift(pred_test, 1, cval=0)
            elif type_of_change == 'pct_change':
                plot_df_train['pred_to_plot'] = Y_train[plotted_feature] * (1 + pred_train)
                plot_df_test['pred_to_plot'] = Y_test[plotted_feature]  * (1 + pred_test)
                for e in pred_test:
                    temps_val = reconstitution[-1] * (1 + e)
                    reconstitution.append(temps_val if temps_val >= 0 else 0)
                plot_df_test['reconstitution'] = reconstitution[1:]
            plot_df_train['pred_to_plot'].clip(0, inplace=True)
            plot_df_test['pred_to_plot'].clip(0, inplace=True)
        plot_df_train['abs_error'] = (plot_df_train['pred_to_plot'] - Y_train[plotted_feature]).abs()
        plot_df_train['pct_error'] = plot_df_train['abs_error'] / Y_train[plotted_feature].abs()
        plot_df_test['abs_error'] = (plot_df_test['pred_to_plot'] - Y_test[plotted_feature]).abs()
        plot_df_test['pct_error'] = plot_df_test['abs_error'] / Y_test[plotted_feature].abs()
        return plot_df_train, plot_df_test

def subplot_model(grid_position,
                  plot_df_train, 
                  plot_df_test, 
                  date_col, 
                  plotted_feature,
                  plot_metrics, 
                  model, 
                  model_params,
                  ncols,
                  text_font_size
                 ):
    """
    Plot the training and test model in the subplot, with errors and title including model name, params and metrics 
    Input: grid_position: tuple with fig, grid, and positions to be occupied by this subplot
            plot_df_train, plot_df_test: outputs from compute_series_to_plot
            date_col, plotted_feature, plot_metrics: same as in plot_model
            model, model_params: current model an params which predictions are to be plotted
            ncols: nb of columns (to handle title width)
            text_font_size: font size for all text computed in visualize_basis based on ncols
    Ouput: metrics on test set for current model
    """
    # define grid position for the subplots
    fig, grid, top, bottom, horiz = grid_position
    ax_main = fig.add_subplot(grid[top:bottom - 1, horiz])
    ax_bottom = fig.add_subplot(grid[bottom - 1:bottom + 1, horiz])
    
    # plot model training and test values
    ax_main.plot(plot_df_train[date_col], plot_df_train[plotted_feature], label="Y_train")
    ax_main.plot(plot_df_train[date_col], plot_df_train['pred_to_plot'], label="Training preds",
                    color=sns.color_palette()[0], alpha=0.5)
    ax_main.plot(plot_df_test[date_col], plot_df_test[plotted_feature], label="Y_test")
    ax_main.plot(plot_df_test[date_col], plot_df_test['pred_to_plot'], label="Test preds")
    try:  # if reconstitution columns in df_test_plot, plot it
        ax_main.plot(plot_df_test[date_col], plot_df_test['reconstitution'], label="Reconstitution",
                    color=sns.color_palette()[2], alpha=0.5)
    except: pass
    ax_main.set_ylabel(plotted_feature.replace("_", " "), fontsize=text_font_size + 2)
    ax_main.legend(loc='best', ncol=1, fontsize=text_font_size)
    
    # abs error subplot below
    col_abs = sns.color_palette()[3]
    ax_bottom.plot(plot_df_train[date_col], plot_df_train['abs_error'],
                   label="Abs error - training", color=col_abs)
    ax_bottom.plot(plot_df_test[date_col], plot_df_test['abs_error'],
                   label="Abs error - test", color=col_abs)
    ax_bottom.set_ylabel("Abs error", fontsize=text_font_size + 2)
    handles_1, _ = ax_bottom.get_legend_handles_labels()
    
    # percentage error subplot below
    ax_bottom_2 = ax_bottom.twinx()
    col_pct = sns.color_palette()[4]
    ax_bottom_2.plot(plot_df_train[date_col], plot_df_train['pct_error'],
                     label="Pct error - training", color=col_pct, alpha=0.8)
    ax_bottom_2.plot(plot_df_test[date_col], plot_df_test['pct_error'],
                     label="Pct error - test", color=col_pct, alpha=0.8)
    ax_bottom_2.set_ylabel("% error (log)", fontsize=text_font_size + 2)
    ax_bottom_2.set_yscale('log')
    ax_bottom_2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax_bottom_2.set_ylim(bottom = 1e-2)
    ax_bottom_2.grid(linestyle="--", color='grey', alpha=0.8)
    handles_2, _ = ax_bottom_2.get_legend_handles_labels()
    ax_bottom_2.legend(
        [handles_1[0], handles_2[0]],
        ["Absolute error", "Percentage error"],
        ncol=1, fontsize=text_font_size, loc='best'
    )

    # vertical line separating training and test set
    x_line = plot_df_train[date_col].iloc[-1]
    ax_main.axvline(x=x_line,
        color='r', alpha=0.5, linestyle='--', linewidth=0.7)
    bottom, top = ax_main.get_ylim()
    y_txt = (top  - bottom) * 0.95 + bottom
    ax_main.text(x=x_line, y=y_txt, s="  Test", va='center' ,ha='left',
        color='r', alpha=0.5, fontsize=text_font_size - 2)
    ax_main.text(x=x_line, y=y_txt, s="Training  ", va='center' ,ha='right',
        color='r', alpha=0.5, fontsize=text_font_size - 2)
    ax_bottom.axvline(x=x_line,
        color='r', alpha=0.5, linestyle='--', linewidth=0.7)
    
    # compute metrics
    if is_regressor(model):
        metrics_test = regression_metrics(plot_df_test[plotted_feature],
                                        plot_df_test['pred_to_plot'],
                                        print_metrics=False
                                        )
    elif is_classifier(model):
        metrics_test = classification_metrics(plot_df_test[plotted_feature],
                                        plot_df_test['pred_to_plot'],
                                        plot_df_test['Y_scores'],
                                        print_metrics=False
                                        )
    else: 
        metrics_test = {}

    # title of subplot containing model type, non-default params and reg/classif metrics
    title_str = 'Model: ' + str(model.__class__.__name__) + '\n'
    if  model_params is not None and len(model_params) > 0:
        title_str += ('Params: ' + str(model_params))
    title_str += '\n'
    if plot_metrics:
        list_str = []
        if is_regressor(model) and 'reconstitution' in plot_df_test.columns:
            rr2 = metrics.r2_score(plot_df_test[plotted_feature], plot_df_test['reconstitution'])
            list_str += [r'Reconstitution $R^2$' + ': {:.4f}'.format(rr2)]
        list_str += [key + ': {:.4f}'.format(value) for key, value in metrics_test.items()]
        metrics_str = []
        for i in range(ncols):  # handle metrics line width in subplot as a function of ncols
            idx = len(list_str) // ncols
            metrics_str += ['  |  '.join(list_str[i * idx: (i + 1) * idx])]
        metrics_str = '\n'.join(metrics_str)
    ax_main.set_title(title_str + metrics_str, fontsize=text_font_size)
    return metrics_test

def regression_metrics(Y_test, Y_pred, print_metrics=True, model_name="", return_series=False):
    """
    Compute several different regression metrics (MAE, RMSE, MAPE, ...) from Y_test and predictions
    """
    diff = Y_test - Y_pred
    local_metrics = {
        r'Test $R^2$': metrics.r2_score(Y_test, Y_pred),
        'ME': np.mean(diff),
        'MAE': np.mean(np.abs(diff)),
        'RMSE': np.sqrt(np.mean(diff**2)),
        'MPE': np.mean(diff / np.mean(Y_test)),
        'MAPE': np.mean(np.abs(diff / Y_test)),
        'MASE': np.mean(np.abs(diff / np.mean(np.abs(np.diff(Y_test))))),
    }
    if print_metrics:
        metrics_names = {
            r'Test $R^2$': r'Test $R^2$',
            'ME': 'Mean error',
            'MAE': 'Mean absolute error',
            'RMSE': 'Root mean squared error',
            'MPE': 'Mean percentage error',
            'MAPE': 'Mean absolute percentage error',
            'MASE': 'Mean absolute scaled error'
        }
        print(" Regression metrics ".center(120, "-"))
        list_str = [metrics_names[m] + ": {:.3f}%".format(100 * local_metrics[m]) for m in local_metrics.keys()]
        print('\n'.join(list_str))
    if return_series:
        series = {
            'Absolute error': diff,
            'Percentage error': diff / Y_test 
        }
        return local_metrics, series
#     reg_metrics = dict(zip(reg_metrics_names.values(), reg_metrics.values()))
    return local_metrics


def classification_metrics(Y_test, Y_pred, Y_scores, labels=None, print_metrics=True, model_name="", return_series=False):
    """
    Compute several different clasification metrics (accuracy, F1 score, AUC, ...) from Y_test and predictions
    """
    Y_scores = Y_scores.to_list()
    if labels is None:
        n_class = Y_scores[0].shape[0]
        labels = np.arange(- (n_class//2), n_class//2 + 1, 1.)
    # ovo and labels needed as Y_test does not always contain all classes
    local_metrics = {
        'Acc': metrics.accuracy_score(Y_test, Y_pred),
        'Precision': metrics.precision_score(Y_test, Y_pred, average='weighted'),
        'Recall': metrics.recall_score(Y_test, Y_pred, average='weighted'),
        'F1': metrics.f1_score(Y_test, Y_pred, average='weighted'),
        'ROCAUC': metrics.roc_auc_score(Y_test, Y_scores, multi_class='ovo', average='weighted', labels=labels),
        'LOGLOSS': metrics.log_loss(Y_test, Y_scores, labels=labels),
    }
    if print_metrics:
        metrics_names = {
            'Acc': 'Accuracy',
            'Precision': 'Precision score',
            'Recall': 'Recall score',
            'F1': 'F1-score',
            'ROCAUC': 'Area under the ROC curve',
            'LOGLOSS': 'Cross-entropy loss',
        }
        print(" Classification metrics {:s} ".format(model_name).center(120, "-"))
        list_str = [reg_metrics_names[m] + ": {:.3f}%".format(100 * local_metrics[m]) for m in local_metrics.keys()]
        print('\n'.join(list_str))
    if return_series:
        series = {
            # 'Absolute error': diff,
            # 'Percentage error': diff / Y_test 
        }
        return local_metrics, series
    return local_metrics
  
