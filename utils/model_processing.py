import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from decimal import Decimal
import itertools

from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit

SEED = 10

def time_series_split_cv(X_train,
                         Y_train,
                         estimator,
                         n_splits=5,
                         params_cv={}  # same format as in sklearn GridSearchCV
                        ):
    """
    Cross-validation from timeseries for all estimators from sklearn
    Input: X_train, Y_train, estimator from sklearn, n_splits=k for k-folds cv, params grid to test (GridSearch only)
    Output: Dataframe with columns params (dict type), # fold, training score and validation score 
    """
    i = 1
    scores = []
    iterator = TimeSeriesSplit(n_splits=n_splits).split(X_train)
    params_list = list(itertools.product(*params_cv.values()))
    params_names = params_cv.keys()
    params_list = [dict(zip(params_names, e)) for e in params_list]
    n_candidates = len(params_list)
    print("\n%s" % " {:d}-folds Cross-validation starting ".format(n_splits).center(120, "-"))
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
            scores.append([params,
                          i,
                          model_cv.score(X_tr, y_tr),
                          model_cv.score(X_val, y_val)
                         ])
        i += 1
    results_cv = pd.DataFrame(scores, columns=['params', 'fold', 'training score' ,'validation score'])
    return results_cv


def plot_model_and_metrics(X_train, X_test, Y_train, Y_test, predicted_feature_name, model):
    """
    Give training and testing score, plot training and test set + training and test predictions
    Input: X_train, X_test, Y_train, Y_test 
        predicted_feature_name: column of Y to be predicted
        model: estimator already fitted to the train set
    Output: training score and test score
    """
    training_score = model.score(X_train, Y_train[predicted_feature_name])
    test_score = model.score(X_test, Y_test[predicted_feature_name])
    print('Training score:', training_score)
    print('Test score:', test_score)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    plt.figure(figsize=(16, 6))
    plt.plot(Y_train['Date'], Y_train[predicted_feature_name], label='Y_train')
    plt.plot(Y_train['Date'], pd.DataFrame(pred_train), label='Predictions - training')
    plt.plot(Y_test['Date'], Y_test[predicted_feature_name], label='Y_test')
    plt.plot(Y_test['Date'], pd.DataFrame(pred_test), label='Predictions - test')
    plt.title("Training and Test performances of model {:s} for {:s}".format(str(model.__class__.__name__),
                                                                            predicted_feature_name.replace("_", " ")
                                                                            ))
        
    plt.legend()
    plt.show()
    return training_score, test_score


def regression_metrics(model, model_name, print_metrics=True):
    return 


def classification_metrics(confusion_matrix, model_name="", print_metrics=True):
    """
    Compute TPR, FPR and accuracy for a given confusion matrix. Possible to print them
    """
    TN=confusion_matrix[0][0]
    FN=confusion_matrix[1][0]
    TP=confusion_matrix[1][1]
    FP=confusion_matrix[0][1]
    acc = (TN + TP) / (TN + FN + TP + FP)
    TPR = TP / (TP+FN)
    FPR = FP/(FP+TN)
    if print_metrics:
        print(model_name + " Accuracy {:.2f}%".format(100 * acc))
        print("\n" + model_name + " TPR {:.2f}%".format(100*TPR))
        print("\n" + model_name + " FPR {:.2f}%".format(100*FPR))
    return acc, TPR, FPR


def plot_confusion_matrix(Y_test, Y_pred, labels=["Positive, Negative"], model_name="", predicted_feature_name=""):
    """
    Input: predcited features test set, predictions, labels for classes, model_name for title, name of the predicted feature
    Plot: confusion matrix
    Output: confusion matrix
    """
    cm = metrics.confusion_matrix(Y_test, Y_pred)   
    df_cm = pd.DataFrame(cm,
                      index = labels,
                      columns=labels
                     )
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.title('Confusion Matrix for ' + model_name, fontweight="bold", fontsize=15)
    plt.ylabel('True ' + predicted_feature_name, fontsize=12)
    plt.yticks([0.2, 1.2], labels)
    plt.xlabel('Predicted ' + predicted_feature_name, fontsize=12)
    return cm


def plot_roc_curve(X_test, Y_test, model, model_name=""):
    """
    Input: X_test, Y_test
        the classification model for which the orc curve is needed
        the model_name for title
    Plot: roc curve
    Output: None 
    """
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(Y_test, probs)
    auc = metrics.auc(fpr, tpr)
    
    sns.set()
    plt.figure(figsize=(8,6))
    # plot baseline
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    # plot model roc curve
    plt.plot(fpr,
             tpr, 
             label="AUC={:.2f}".format(auc)
            )

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("FPR")
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("TPR")
    plt.title('ROC Curve Analysis for ' + model_name, fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    
