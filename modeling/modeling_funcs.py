# -*- coding: utf-8 -*-
"""
Modeling Functions

06/02/2019
Jared Berry
"""

import warnings
warnings.filterwarnings('always')

# Data structures
import pickle
import numpy as np
import pandas as pd

# Model selection
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

# Evaluation
from sklearn import metrics

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

# Quality of life
import time

def prepare_model_structures(X, y, holdout, labeled=False, ema_gamma=1):
    """
    Given a dataframe of features, a target array, and
    holdout test set; label and smooth if necessary.
    Returns a tuple of prepared structures for modeling
    """
    
    # Convert to NumPy arrays - store feature names
    feature_names = X.columns.tolist()
    features = np.array(X)
    test_features = np.array(holdout)
    targets = y.copy()
    
    if labeled:
        targets_smoothed = targets.copy()
        orig_targets = targets.copy()
    else:
        # Compute EMA smoothing of target prior to constructing classes
        EMA = 0
        gamma_ = ema_gamma
        for ti in range(len(targets)):
            EMA = gamma_*targets[ti] + (1-gamma_)*EMA
            targets[ti] = EMA  

        targets_smoothed = np.array(np.where(targets > 0, 1, 0), dtype='int')
        orig_targets = np.array(np.where(y.copy() > 0, 1, 0), dtype='int')
        print("\n{} targets changed by smoothing.".format(np.sum(targets_smoothed != orig_targets)))
        
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
        
    return (features, feature_names, feature_importance_values, \
            test_features, test_predictions, out_of_fold, \
            targets_smoothed, orig_targets)
    
def benchmark_target(target, groups, grouping_var = 'ticker'):
    """
    Benchmark classification metrics for the target
    against a one-class target and random-walk per
    time-series literature; allows a grouping object
    to conduct the random-walk shifting at the entity
    level.
    Returns nothing; prints benchmark statistics.
    """
    
    # Single class benchmark
    one_class = np.ones(len(target), dtype='int')
    
    # Shift to create random walks, by ticker if necessary
    if groups.empty:
        rw_classes = np.array(pd.Series(target).shift(1))  
    else:
        groups_w_target = groups.copy(deep=True)
        groups_w_target['target'] = target
        rw_classes = (np.array(groups_w_target
                               .groupby(grouping_var)['target']
                               .shift(1)))        
        
    msg_rws = np.isnan(rw_classes)
    rw_class = np.array(rw_classes[~msg_rws], dtype='int')
    target_rw = np.array(target[~msg_rws], dtype='int')
    
    # Compute and report metrics
    one_class_acc = 100*np.round(np.mean(one_class == target), 2)
    print("="*79)
    print("Baseline, one-class accuracy is: {}%".format(one_class_acc))
    print("Classification report for one-class predictor:")
    print(metrics.classification_report(target, one_class))
    
    rw_class_acc = 100*np.round(np.mean(rw_class == target_rw), 2)
    print("Baseline, random-walk accuracy is: {}%".format(rw_class_acc))
    print("Classification report for random-walk predictor:")
    print(metrics.classification_report(target_rw, rw_class)) 
    print("="*79)     
    
def PanelSplit(n_folds, groups, grouping_var='ticker'):
    """
    Function to generate time series splits of a panel, provided
    a number of folds, and an indexable dataframe to create groups.
    Returns a generator object for compliance with sci-kit learn API.
    """
    by_ticker_index = (groups.groupby(grouping_var)
                       .apply(lambda x: x.reset_index(drop=True))
                       .drop(grouping_var, axis=1)
                       .reset_index()
                       .rename({'level_1':'tsidx'}, axis=1)
                       )
    
    ticker_range = by_ticker_index['tsidx'].unique().tolist()
    ticker_range = sorted(ticker_range)
    
    splits = TimeSeriesSplit(n_splits=n_folds)
    
    for train_indices, valid_indices in splits.split(ticker_range):
        panel_train_indices = by_ticker_index[by_ticker_index['tsidx'].isin(train_indices)].index.tolist()
        panel_valid_indices = by_ticker_index[by_ticker_index['tsidx'].isin(valid_indices)].index.tolist()
        yield panel_train_indices, panel_valid_indices
        
def WindowSplit(window, groups, panel):    
    """
    Function to generate windowed time series splits of a panel, provided
    a number of folds, and an indexable dataframe to create groups.
    Returns a generator object for compliance with sci-kit learn API.
    """    
    wparams = window.split(':')
    wtrain = int(wparams[0])
    wvalid = int(wparams[1])
    witer = int(wparams[2])
    
    by_ticker_index = (groups.groupby('ticker')
                       .apply(lambda x: x.reset_index(drop=True))
                       .drop('ticker', axis=1)
                       .reset_index()
                       .rename({'level_1':'tsidx'}, axis=1)
                       )
    
    ticker_range = by_ticker_index['tsidx'].unique().tolist()
    ticker_range = sorted(ticker_range)
    
    stop = 0
    start = (max(ticker_range) % witer) + 1
    if panel:
        while stop < max(ticker_range):
            train_indices = np.arange(start, start + wtrain).tolist()
            valid_indices = np.arange(start + wtrain, start + wtrain + wvalid).tolist()
            stop = max(valid_indices)
            start += witer
            yield train_indices, valid_indices
    else:
        while stop < max(ticker_range):
            train_indices = np.arange(start, start + wtrain).tolist()
            valid_indices = np.arange(start + wtrain, start + wtrain + wvalid).tolist()
            panel_train_indices = by_ticker_index[by_ticker_index['tsidx'].isin(train_indices)].index.tolist()
            panel_valid_indices = by_ticker_index[by_ticker_index['tsidx'].isin(valid_indices)].index.tolist()
            stop = max(valid_indices)
            start += witer
            yield panel_train_indices, panel_valid_indices
            
def instantiate_splits(X, n_splits, groups, cv_method='ts'):
    """
    Create one of three cross-nation split
    generator objects based on specified nation
    method.
    Returns two sets of splits for use in training
    and GridSearchCV, if necessary
    """
    
    if cv_method == "panel":
        splits = PanelSplit(n_folds=n_splits, groups=groups)
        search_splits = PanelSplit(n_folds=n_splits, groups=groups)
    elif cv_method == "ts":
        splits = TimeSeriesSplit(n_splits=n_splits).split(X)
        search_splits = TimeSeriesSplit(n_splits=n_splits).split(X)
    elif cv_method == "kfold":
        splits = KFold(n_splits=n_splits).split(X)
        search_splits = KFold(n_splits=n_splits).split(X)
        
    return splits, search_splits

def discrimination_threshold_search(predicted, expected, search_range=[0.25, 0.75], step=0.05, 
                                    metric=metrics.precision_score):
    """
    Search over a specified range of discrimination 
    thresholds and maximize relative to a specified metric.
    Returns the optimum threshold
    """
    
    thresholds = list(np.arange(search_range[0], search_range[1], step))
    preds = [[1 if y >= t else 0 for y in predicted] for t in thresholds]
    scores_by_threshold = [metrics(expected, p) for p in preds]
    optimum = thresholds[scores_by_threshold.index(max(scores_by_threshold))]
    
    return(optimum)
    
def fit_sklearn_classifier(X, y, holdout, ticker, ema_gamma, n_splits, model, label, param_search={}, export=False,
                           cv_method="ts", labeled=False, groups=pd.DataFrame(), threshold_search=False, **kwargs):
    """
    Flexible function for fitting any number of sci-kit learn
    classifiers, with optional grid search.
    """

    start = time.time()
    
    # Prepare modeling structures - unpack
    features, feature_names, feature_importance_values, test_features, \
    test_predictions, out_of_fold, targets_smoothed, orig_targets = \
    prepare_model_structures(X, y, holdout, labeled, ema_gamma)
    
    # Compute some baselines
    benchmark_target(targets_smoothed, groups)

    # Instantiate cross-validation splitting generators
    splits, search_splits = instantiate_splits(X, n_splits, groups, cv_method)
    
    # Dictionary of lists for recording validation and training scores
    scores = {'precision':[], 'recall':[], 'accuracy':[], 'f1':[]}
    
    # Perform a grid-search on the provided parameters to determine best options
    if param_search:
        print("Performing grid search for hyperparameter tuning")
        gsearch = GridSearchCV(estimator=model(**kwargs), 
                                  cv=search_splits,
                                  param_grid=param_search)

        # Fit to extract best parameters later
        gsearch_model = gsearch.fit(features, targets_smoothed)

    split_counter = 1
    for train_indices, test_indices in splits:
        print("Training model on validation split #{}".format(split_counter))
        
        # Train/test split
        train_features, train_targets = features[train_indices], targets_smoothed[train_indices]
        test_features, expected = features[test_indices], targets_smoothed[test_indices]
        
        # Generate a model given the optimal parameters established in grid search
        if param_search:
            estimator = model(**gsearch_model.best_params_)
        else:
            estimator = model(**kwargs)
            
        # Train the estimator; fit 
        estimator.fit(train_features, train_targets)
        probs = [p[1] for p in estimator.predict_proba(test_features)]

        # Dynamic classification threshold selection
        if threshold_search:
            opt_threshold = discrimination_threshold_search(probs, expected)
        else:
            opt_threshold = 0.5
            
        predicted = [1 if y >= opt_threshold else 0 for y in probs]

        # Append scores to the tracker
        scores['precision'].append(metrics.precision_score(expected, predicted, average="weighted"))
        scores['recall'].append(metrics.recall_score(expected, predicted, average="weighted"))
        scores['accuracy'].append(metrics.accuracy_score(expected, predicted))
        scores['f1'].append(metrics.f1_score(expected, predicted, average="weighted"))
        
        # Store variable importances 
        if model in [RandomForestClassifier, GradientBoostingClassifier]:
            feature_importance_values += estimator.feature_importances_ / n_splits
            
        # Iterate counter
        split_counter += 1
            
    # Properly format feature importances
    named_importances = list(zip(feature_names, feature_importance_values))
    sorted_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)
    
    # Fit on full sample 
    if param_search:
        estimator = model(**gsearch_model.best_params_)
    else:
        estimator = model(**kwargs)
        
    estimator.fit(features, targets_smoothed)
    probs = [p[1] for p in estimator.predict_proba(test_features)]
    
    if threshold_search:
        opt_threshold = discrimination_threshold_search(probs, expected)
    else:
        opt_threshold = 0.5
        
    predicted = [1 if y >= opt_threshold else 0 for y in probs]

    # Store values for later reporting/use in app
    evals = {'precision':np.mean(scores['precision']), 
             'recall':np.mean(scores['recall']), 
             'accuracy':np.mean(scores['accuracy']), 
             'f1':np.mean(scores['f1']),
             'probabilities':probs,
             'predictions':predicted,
             'importances':sorted_importances
             }
    
    # Report
    print("Build, hyperparameter selection, and validation of {} took {:0.3f} seconds\n".format(label, time.time()-start))
    print("Hyperparameters are as follows:")
    if param_search:
        for key in gsearch_model.best_params_.keys():
            print("{}: {}\n".format(key, gsearch_model.best_params_[key]))
    print("Validation scores are as follows:")
    print(pd.DataFrame(scores).mean())
    
    # Output the evals dictionary
    if export:
        outpath = "{}_{}.pickle".format(label.tolower().replace(" ", "_"), ticker.tolower())
        with open(outpath, 'wb') as f:
            pickle.dump(evals, f)
            
def fit_lgbm_classifier(X, y, holdout, ticker="", ema_gamma=1, n_splits=12, label="LGBM Classifier", param_search={}, 
                        export=False, cv_method="ts", labeled=False, groups=pd.DataFrame(), threshold_search=False, **kwargs):
    """
    Flexible function for fitting LightGBM
    classifiers, with optional grid search.
    """    

    start = time.time()
    
    # Prepare modeling structures - unpack
    features, feature_names, feature_importance_values, test_features, \
    test_predictions, out_of_fold, targets_smoothed, orig_targets = \
    prepare_model_structures(X, y, holdout, labeled, ema_gamma)
    
    # Compute some baselines
    benchmark_target(targets_smoothed, groups)

    # Instantiate cross-validation splitting generators
    splits, search_splits = instantiate_splits(X, n_splits, groups, cv_method)
    
    # Dictionary of lists for recording validation and training scores
    scores = {'precision':[], 'recall':[], 'accuracy':[], 'f1':[]}
    
    # Lists for recording validation and training scores
    test_scores = []
    train_scores = []
    test_accs = []
    train_accs = [] 
    
    # Perform a grid-search on the provided parameters to determine best options
    if param_search:
        print("Performing grid search for hyperparameter tuning")
        gsearch = GridSearchCV(estimator=LGBMClassifier(**kwargs), 
                                  cv=search_splits,
                                  param_grid=param_search)

        # Fit to extract best parameters later
        gsearch_model = gsearch.fit(features, targets_smoothed)

    split_counter = 1
    for train_indices, test_indices in splits: 
        print("Training model on validation split #{}".format(split_counter))
              
        # Train/test split
        train_features, train_targets = features[train_indices], targets_smoothed[train_indices]
        valid_features, expected = features[test_indices], targets_smoothed[test_indices]

        # Generate a bst model given the optimal parameters established in grid search
        if param_search:
            bst = LGBMClassifier(**gsearch_model.best_params_)
        else:
            bst = LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                 class_weight = 'balanced', learning_rate = 0.01,
                                 ##max_bin = 25, num_leaves = 25, max_depth = 1,
                                 reg_alpha = 0.1, reg_lambda = 0.1, 
                                 subsample = 0.8, random_state = 101
                                )

        # Train the bst
        bst.fit(train_features, train_targets, eval_metric = ['auc', 'binary_error'],
                eval_set = [(valid_features, expected), (train_features, train_targets)],
                eval_names = ['test', 'train'], early_stopping_rounds = 100, verbose = 0)

        # Record the best iteration
        best_iteration = bst.best_iteration_

        # Record the feature importances
        feature_importance_values += bst.feature_importances_ / n_splits
        
        # Make predictions
        #test_predictions += bst.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / n_splits

        # Record the out of fold predictions
        out_of_fold[test_indices] = bst.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

        # Record the best score
        test_score = bst.best_score_['test']['auc']
        train_score = bst.best_score_['train']['auc']
        test_acc = bst.best_score_['test']['binary_error']
        train_acc = bst.best_score_['train']['binary_error']
        
        test_scores.append(test_score)
        train_scores.append(train_score)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        
        split_counter += 1 
        
    # Properly format feature importances
    named_importances = list(zip(feature_names, feature_importance_values))
    sorted_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)
        
    # Set up an exportable dictionary with results from the model
    results = {
        'train_auc':train_scores,
        'train_acc':train_acc,
        'validation_auc':test_scores,
        'validation_accuracy':test_acc,
        'test_preds':out_of_fold,
        'feature_importances':sorted_importances,
        'test_predictions': None #test_predictions
    }
    
    # Output the results dictionary
    if export:
        outpath = "lgbc_{}.pickle".format(ticker.tolower())
        with open(outpath, 'wb') as f:
            pickle.dump(results, f)
    
    print("Build, hyperparameter selection, and validation of {} took {:0.3f} seconds\n".format(label, time.time()-start))
    print("Average AUC across {} splits: {}".format(n_splits, np.mean(test_scores)))
    print("Average accuracy across {} splits: {}%".format(n_splits, 100*np.round((1-np.mean(test_acc)),2)))
    for i in range(6):
        print(sorted_importances[i])