# -*- coding: utf-8 -*-
"""
Modeling Functions

06/02/2019
Jared Berry
"""
# Quality of life
import time
import re
import pickle

# Data structures
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

def slugify(value):
    """
    Converts to ASCII. Converts spaces to underscores. Removes characters that
    aren't alphanumerics, underscores, or hyphens. Converts to lowercase.
    Also strips leading and trailing whitespace.
    """
    value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)

def prepare_model_structures(X, y, holdout, labeled=False, ema_gamma=1):
    """
    Given a dataframe of features, a target array, and
    holdout test set; label and smooth if necessary.
    Returns a tuple of prepared structures for modeling
    """

    # Convert to NumPy arrays - store feature names
    feature_names = X.columns.tolist()
    features = np.array(X)
    holdout_features = np.array(holdout)
    targets = y.copy()

    if labeled:
        targets_smoothed = targets.copy()
        orig_targets = targets.copy()
    else:
        # Compute ema smoothing of target prior to constructing classes
        ema = 0
        gamma_ = ema_gamma
        for t_i, _ in enumerate(targets):
            ema = gamma_*targets[t_i] + (1-gamma_)*ema
            targets[t_i] = ema

        targets_smoothed = np.array(np.where(targets > 0, 1, 0), dtype='int')
        orig_targets = np.array(np.where(y.copy() > 0, 1, 0), dtype='int')
        print("\n{} targets changed by smoothing.".format(np.sum(targets_smoothed != orig_targets)))

    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    # Empty array for out of fold validation predictions, and thresholds
    out_of_fold = np.zeros(features.shape[0])
    thresholds = np.zeros(features.shape[0])
    predicted_out_of_fold = np.zeros(features.shape[0])
    split_nums = np.zeros(features.shape[0])

    # Empty array for test predictions
    test_predictions = np.zeros(holdout_features.shape[0])

    return (features, feature_names, feature_importance_values, \
            holdout_features, test_predictions, out_of_fold, \
            targets_smoothed, thresholds, predicted_out_of_fold, \
            split_nums)

def benchmark_target(target, groups, grouping_var='ticker'):
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

def panel_split(n_folds, groups, grouping_var='date_of_transaction'):
    """
    Function to generate time series splits of a panel, provided
    a number of folds, and an indexable dataframe to create groups.
    Returns a generator object for compliance with sci-kit learn API.
    """
    date_idx = (groups[[grouping_var]]
                .drop_duplicates()
                .sort_values(grouping_var)
                .reset_index()
                .rename({'index':'tsidx'}, axis=1))

    by_ticker_index = groups.reset_index().rename({'index':'panel_index'}, axis=1)
    by_ticker_index = (pd.merge(by_ticker_index, date_idx, on=grouping_var)
                       .sort_values('panel_index')
                       .set_index('panel_index'))

    ticker_range = sorted(by_ticker_index['tsidx'].unique().tolist())

    splits = TimeSeriesSplit(n_splits=n_folds)

    for train_indices, test_indices in splits.split(ticker_range):
        panel_train_indices = (by_ticker_index[by_ticker_index['tsidx'].isin(train_indices)]
                               .index
                               .tolist())
        panel_test_indices = (by_ticker_index[by_ticker_index['tsidx'].isin(test_indices)]
                              .index
                              .tolist())
        yield panel_train_indices, panel_test_indices

def n_day_ahead_split(indexer, train=252, test=21, window=False,
                      grouping_var='date_of_transaction'):
    """
    Function to generate time series splits of a panel or time series, provided
    a dedicated minimum for the training sample and a dedicated testing window.
    Default is to use a minimum of a year's worth of data with the month ahead
    horizon for testing consistent with most constructed targets.
    Returns a generator object for compliance with sci-kit learn API.
    """
    if isinstance(indexer, pd.DataFrame):
        date_idx = (indexer[[grouping_var]]
                    .drop_duplicates()
                    .sort_values(grouping_var)
                    .reset_index()
                    .rename({'index':'tsidx'}, axis=1))

        by_ticker_index = indexer.reset_index().rename({'index':'panel_index'}, axis=1)
        by_ticker_index = (pd.merge(by_ticker_index, date_idx, on=grouping_var)
                           .sort_values('panel_index')
                           .set_index('panel_index'))

        ticker_range = sorted(by_ticker_index['tsidx'].unique().tolist())
        buffer = max(ticker_range) % test

        end = train + buffer
        start = 0
        while end < max(ticker_range):
            train_indices = (by_ticker_index[by_ticker_index['tsidx'].isin(np.arange(start, end))]
                             .index.tolist())
            test_indices = (by_ticker_index[by_ticker_index['tsidx']
                                            .isin(np.arange(end, (end+test)))]
                            .index
                            .tolist())

            end += test
            if window:
                start += test

            yield train_indices, test_indices

    else:
        buffer = indexer % test
        end = train + buffer
        start = 0
        while end < indexer:
            train_indices = np.arange(start, end).tolist()
            test_indices = np.arange(end, (end+test)).tolist()
            end += test
            if window:
                start += test

            yield train_indices, test_indices

def instantiate_splits(X, n_splits, groups, cv_method='ts'):
    """
    Create one of several cross-validation split generator objects based on
    specified validation method.
    Returns two sets of splits for use in training and GridSearchCV.
    """

    if cv_method == "panel":
        splits = panel_split(n_folds=n_splits, groups=groups)
        search_splits = panel_split(n_folds=n_splits, groups=groups)
    elif cv_method == "ts":
        splits = TimeSeriesSplit(n_splits=n_splits).split(X)
        search_splits = TimeSeriesSplit(n_splits=n_splits).split(X)
    elif cv_method == "kfold":
        splits = KFold(n_splits=n_splits).split(X)
        search_splits = KFold(n_splits=n_splits).split(X)
    elif cv_method == "tsrecur":
        splits = n_day_ahead_split(X.shape[0], window=False)
        search_splits = n_day_ahead_split(X.shape[0], window=False)
    elif cv_method == "panelrecur":
        splits = n_day_ahead_split(indexer=groups, window=False)
        search_splits = n_day_ahead_split(indexer=groups, window=False)
    elif cv_method == "tswindow":
        splits = n_day_ahead_split(X.shape[0], window=True)
        search_splits = n_day_ahead_split(X.shape[0], window=True)
    elif cv_method == "panelwindow":
        splits = n_day_ahead_split(indexer=groups, window=True)
        search_splits = n_day_ahead_split(indexer=groups, window=True)

    return splits, search_splits

def discrimination_threshold_search(predicted, expected, search_min=0.25, search_max=0.75,
                                    metric=metrics.precision_score):
    """
    Search over a specified range of discrimination
    thresholds and maximize relative to a specified metric.
    Returns the optimum threshold
    """

    thresholds = list(np.arange(search_min, search_max, 0.05))
    preds = [[1 if y >= t else 0 for y in predicted] for t in thresholds]
    scores_by_threshold = [metric(expected, p) for p in preds]
    optimum = thresholds[scores_by_threshold.index(max(scores_by_threshold))]

    return optimum

def fit_sklearn_classifier(X, y, holdout, ticker, ema_gamma, n_splits, model, label,
                           param_search=None, cv_method="ts", labeled=False,
                           groups=pd.DataFrame(), threshold_search=False,
                           smooth_train_targets=False, ema_gamma_train=1,
                           benchmarks=False, holdout_method="distributed",
                           export=False, **kwargs):
    """
    Flexible function for fitting any number of sci-kit learn
    classifiers, with optional grid search.
    """

    start = time.time()
    if ('recur' or 'window') in cv_method:
        n_splits = 1

    # Prepare modeling structures - unpack
    features, feature_names, feature_importance_values, holdout_features, \
    test_predictions, out_of_fold, targets_smoothed, \
    thresholds, predicted_out_of_fold, split_nums = \
    prepare_model_structures(X, y, holdout, labeled, ema_gamma)

    # Compute some baselines
    if benchmarks:
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

        if smooth_train_targets:
            ema = 0
            gamma_ = ema_gamma_train
            for t_i, _ in enumerate(train_targets):
                ema = gamma_*train_targets[t_i] + (1-gamma_)*ema
                train_targets[t_i] = ema

        # Generate a model given the optimal parameters established in grid search
        if param_search:
            estimator = model(**gsearch_model.best_params_)
        else:
            estimator = model(**kwargs)

        # Train the estimator; fit and store
        estimator.fit(train_features, train_targets)
        probs = [p[1] for p in estimator.predict_proba(test_features)]
        out_of_fold[test_indices] = probs
        split_nums[test_indices] = split_counter

        # Dynamic classification threshold selection
        if threshold_search:
            opt_threshold = discrimination_threshold_search(probs, expected)
            thresholds[test_indices] = opt_threshold
        else:
            opt_threshold = 0.5
            thresholds[test_indices] = opt_threshold

        # Make predictions
        if holdout_method == "distributed":
            holdout_probs = np.array([p[1] for p in estimator.predict_proba(holdout_features)])
            test_predictions += holdout_probs / n_splits

        predicted = [1 if y >= opt_threshold else 0 for y in probs]
        predicted_out_of_fold[test_indices] = predicted

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

    # Adjust distributed metrics
    if ('recur' or 'window') in cv_method:
        feature_importance_values = feature_importance_values / split_counter
        test_predictions = test_predictions / split_counter

    # Properly format feature importances
    named_importances = list(zip(feature_names, feature_importance_values))
    sorted_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)

    # Fit on full sample
    if holdout_method == "full":
        if param_search:
            estimator = model(**gsearch_model.best_params_)
        else:
            estimator = model(**kwargs)

        estimator.fit(features, targets_smoothed)
        holdout_probs = np.array([p[1] for p in estimator.predict_proba(holdout_features)])
        test_predictions += holdout_probs

    # Create a dataframe for model evaluation
    cols = ["split_number", "expected", "predicted_prob", "threshold", "predicted"]
    vals = [split_nums, targets_smoothed, out_of_fold, thresholds, predicted_out_of_fold]
    preds = pd.DataFrame().from_items(zip(cols, vals))
    preds['ticker'] = ticker
    preds['model'] = label

    # Store values for later reporting/use in app
    results = {'preds_df':preds,
               'probabilities':probs,
               'holdout_probs':test_predictions,
               'importances':sorted_importances}

    # Report
    print("Build, hyperparameter selection, and validation of {} took {:0.3f} seconds\n"
          .format(label, time.time()-start))
    print("Hyperparameters are as follows:")
    if param_search:
        for key in gsearch_model.best_params_.keys():
            print("{}: {}\n".format(key, gsearch_model.best_params_[key]))
    print("Validation scores are as follows:")
    print(pd.DataFrame(scores).mean())

    # Export
    if export:
        if not ticker:
            ticker = 'panel'
        outpath = "{}_{}_{}.pickle".format(slugify(label), ticker, cv_method)
        with open(outpath, 'wb') as f:
            pickle.dump(results, f)

    return results

def fit_lgbm_classifier(X, y, holdout, ticker="", ema_gamma=1, n_splits=12,
                        label="LGBM Classifier", param_search=None, cv_method="ts",
                        labeled=False, groups=pd.DataFrame(), threshold_search=False,
                        ema_gamma_train=1, smooth_train_targets=False, benchmarks=False,
                        holdout_method="distributed", export=False, **kwargs):
    """
    Flexible function for fitting LightGBM
    classifiers, with optional grid search.
    """

    start = time.time()
    if ('recur' or 'window') in cv_method:
        n_splits = 1

    # Prepare modeling structures - unpack
    features, feature_names, feature_importance_values, holdout_features, \
    test_predictions, out_of_fold, targets_smoothed, \
    thresholds, predicted_out_of_fold, split_nums = \
    prepare_model_structures(X, y, holdout, labeled, ema_gamma)

    # Compute some baselines
    if benchmarks:
        benchmark_target(targets_smoothed, groups)

    # Instantiate cross-validation splitting generators
    splits, search_splits = instantiate_splits(X, n_splits, groups, cv_method)

    # Lists for recording validation and training scores
    test_scores = []
    train_scores = []

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
        test_features, expected = features[test_indices], targets_smoothed[test_indices]

        if smooth_train_targets:
            ema = 0
            gamma_ = ema_gamma_train
            for t_i, _ in enumerate(train_targets):
                ema = gamma_*train_targets[t_i] + (1-gamma_)*ema
                train_targets[t_i] = ema

        # Generate a bst model given the optimal parameters established in grid search
        if param_search:
            bst = LGBMClassifier(**gsearch_model.best_params_)
        else:
            bst = LGBMClassifier(n_estimators=1000, objective='binary',
                                 class_weight='balanced', learning_rate=0.01,
                                 #max_bin = 25, num_leaves = 25, max_depth = 1,
                                 reg_alpha=0.1, reg_lambda=0.1,
                                 subsample=0.8, random_state=101
                                )

        # Train the bst
        bst.fit(train_features, train_targets, eval_metric=['auc'],
                eval_set=[(test_features, expected), (train_features, train_targets)],
                eval_names=['test', 'train'], early_stopping_rounds=100, verbose=0)

        # Record the best iteration
        best_iteration = bst.best_iteration_

        # Record the feature importances
        feature_importance_values += bst.feature_importances_ / n_splits

        # Make predictions
        if holdout_method == "distributed":
            test_predictions += (bst.predict_proba(holdout_features,
                                                   num_iteration=best_iteration)[:, 1] /
                                 n_splits)

        # Record the out of fold predictions
        out_of_fold[test_indices] = bst.predict_proba(test_features,
                                                      num_iteration=best_iteration)[:, 1]
        split_nums[test_indices] = split_counter

        probs = bst.predict_proba(test_features, num_iteration=best_iteration)[:, 1]

        # Dynamic classification threshold selection
        if threshold_search:
            opt_threshold = discrimination_threshold_search(probs, expected)
            thresholds[test_indices] = opt_threshold
        else:
            opt_threshold = 0.5
            thresholds[test_indices] = opt_threshold

        predicted = np.array([1 if y >= opt_threshold else 0 for y in probs])
        predicted_out_of_fold[test_indices] = predicted

        # Record the best score
        test_score = bst.best_score_['test']['auc']
        train_score = bst.best_score_['train']['auc']

        test_scores.append(test_score)
        train_scores.append(train_score)

        split_counter += 1

    # Adjust distributed metrics
    if ('recur' or 'window') in cv_method:
        feature_importance_values = feature_importance_values / split_counter
        test_predictions = test_predictions / split_counter

    # Properly format feature importances
    named_importances = list(zip(feature_names, feature_importance_values))
    sorted_importances = sorted(named_importances, key=lambda x: x[1], reverse=True)

    # Create a dataframe for model evaluation
    cols = ["split_number", "expected", "predicted_prob", "threshold", "predicted"]
    vals = [split_nums, targets_smoothed, out_of_fold, thresholds, predicted_out_of_fold]
    preds = pd.DataFrame().from_items(zip(cols, vals))
    preds['ticker'] = ticker
    preds['model'] = label

    # Set up an exportable dictionary with results from the model
    results = {
        'train_auc':train_scores,
        'validation_auc':test_scores,
        'preds_df':preds,
        'feature_importances':sorted_importances,
        'holdout_probs':test_predictions}

    print("Build, hyperparameter selection, and validation of {} took {:0.3f} seconds\n"
          .format(label, time.time()-start))
    print("Average AUC across {} splits: {}".format(n_splits, np.mean(test_scores)))
    for i in range(6):
        print(sorted_importances[i])

    # Export
    if export:
        if not ticker:
            ticker = 'panel'
        outpath = "{}_{}_{}.pickle".format(slugify(label), ticker, cv_method)
        with open(outpath, 'wb') as f:
            pickle.dump(results, f)

    return results
