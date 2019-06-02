# -*- coding: utf-8 -*-
"""
Modeling Functions

06/02/2019
Jared Berry
"""
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

        targets_smoothed = np.where(targets > 0, 1, 0)    
        orig_targets = np.where(y.copy() > 0, 1, 0)
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
    if groups:
        groups_w_target = groups.copy(deep=True)
        groups_w_target['target'] = target
        rw_classes = (np.array(groups_w_target
                               .groupby(grouping_var)['target']
                               .shift(1)))
        
    else:
        rw_classes = np.array(pd.Series(target).shift(1))
        
    msg_rws = np.isnan(rw_classes)
    rw_class = np.array(rw_classes[~msg_rws], dtype='int')
    target_rw = np.array(target[~msg_rws], dtype='int')
    
    # Compute and report metrics
    one_class_acc = 100*np.round(np.mean(one_class == target), 2)
    print("Baseline, one-class accuracy is: {}%".format(one_class_acc))
    print("Classification report for one-class predictor:")
    print(metrics.classification_report(target, one_class))
    
    rw_class_acc = 100*np.round(np.mean(rw_class == target_rw), 2)
    print("Baseline, one-class accuracy is: {}%".format(rw_class_acc))
    print("Classification report for one-class predictor:")
    print(metrics.classification_report(target_rw, rw_class))      
    
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
        search_splits = Kfold(n_splits=n_splits).split(X)
        
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