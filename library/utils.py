from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm.notebook import tqdm
import pandas as pd
import copy
from timeit import default_timer
import numpy as np

def get_folds(X, y, n_splits = 5, shuffle = True, random_state = 1234):
    """
    y should be a single label, not a multiclass label
    """
#     print("X", X.shape)
#     print("y", y.shape)
    kf = StratifiedKFold(
        n_splits = n_splits, 
        shuffle = shuffle, 
        random_state = random_state
    )
#     kf.get_n_splits(X)
    
    output = []
#     labels_strat = data.task_1.astype(int).to_numpy()
    for train_idx, val_idx in kf.split(X, y):
        t = (X[train_idx], y[train_idx])
        v = (X[val_idx], y[val_idx])
        output.append((t,v))
    
    return output

def validate(
    X, 
    Y, 
    classifier, 
    n_splits = 5, 
    shuffle = True, 
    random_state = 1234, 
    full = False,
    verbose = True,
    **kwargs
):
    print("Cross-validation process started...")
    start = default_timer()
    results = []
    for j, c in enumerate(Y.columns, 1):
        data = get_folds(X, Y[c], n_splits, shuffle, random_state)
        for i, ((a_train, b_train), (a_test, b_test)) in enumerate(data, 1):
#             fresh_classifier = copy.deepcopy(classifier)
            if verbose:
                print(f"*** column {j} / {len(Y.columns)} ({c}) - fold {i} / {len(data)}")
                print("    training model...")
            classifier.entrenar(a_train, b_train, **kwargs)
            if verbose: 
                print("    generating predictions on the train set...")
            train_predictions = classifier.predecir(a_train)  
            if verbose: 
                print("    generating predictions on the test set...")
            test_predictions = classifier.predecir(a_test)  
            results.append(
                dict(
                    fold = i,
                    column = c,
                    train_accuracy = accuracy_score(b_train, train_predictions),
                    train_precision = precision_score(b_train, train_predictions, zero_division = 0.0),
                    train_recall = recall_score(b_train, train_predictions, zero_division = 0.0),
                    train_f1 = f1_score(b_train, train_predictions, zero_division = 0.0),
                    test_accuracy = accuracy_score(b_test, test_predictions),
                    test_precision = precision_score(b_test, test_predictions, zero_division = 0.0),
                    test_recall = recall_score(b_test, test_predictions, zero_division = 0.0),
                    test_f1 = f1_score(b_test, test_predictions, zero_division = 0.0),
                )
            )
            time = default_timer() - start
            print(f"    Total runtime: {time/60:.2f} minutes")
    results = pd.DataFrame(results)
    if full:
        return results
    else:
        return results.pivot_table(
            index = "column", 
            values = [
                "train_accuracy", "train_precision", "train_recall", "train_f1", 
                "test_accuracy", "test_precision", "test_recall", "test_f1"
            ]
        )

#---------------------------------------------
#--------------MTL - task 2-------------------
#---------------------------------------------
    
def validate_MTL(
    X, 
    Y, 
    classifier, 
    n_splits = 5, 
    shuffle = True, 
    random_state = 1234, 
    full = False,
    verbose = True,
    params = None
):
    columns = params.get('columns')
    print("X", X.shape)
    print("Y", Y.shape)
    print("Cross-validation process started...")
    start = default_timer()
    results = []
    data = get_folds_MTL(X, Y, n_splits, shuffle, random_state)
    for i, ((a_train, b_train), (a_test, b_test)) in enumerate(data, 1):
        if verbose:
            print(f"*** fold {i} / {len(data)}")
            print("    training model...")
        classifier.entrenar(((a_train, b_train), (a_test, b_test)), params)
        if verbose: 
            print("    generating predictions on the train set...")
        train_predictions = classifier.predecir(a_train)  
        if verbose: 
            print("    generating predictions on the test set...")
        test_predictions = classifier.predecir(a_test)  
        for target in range(len(b_train)):
            results.append(
                dict(
                    fold = i,
                    column = columns.get(target),                    
                    train_accuracy = accuracy_score(b_train[target], train_predictions[target]),
                    train_precision = precision_score(b_train[target], train_predictions[target]),
                    train_recall = recall_score(b_train[target], train_predictions[target]),
                    train_f1 = f1_score(b_train[target], train_predictions[target]),
                    test_accuracy = accuracy_score(b_test[target], test_predictions[target]),
                    test_precision = precision_score(b_test[target], test_predictions[target]),
                    test_recall = recall_score(b_test[target], test_predictions[target]),
                    test_f1 = f1_score(b_test[target], test_predictions[target]),
                )
            )
            
        # adding task_1 predictions
        t1_train_predictions = np.logical_or.reduce(train_predictions)
        t1_test_predictions = np.logical_or.reduce(test_predictions)
        t1_b_train = np.logical_or.reduce(b_train)
        t1_b_test = np.logical_or.reduce(b_test)
        print(t1_test_predictions.shape)
        print(t1_b_test.shape)
        results.append(
                dict(
                    fold = i,
                    column = 'task_1',                    
                    train_accuracy = accuracy_score(t1_b_train, t1_train_predictions),
                    train_precision = precision_score(t1_b_train, t1_train_predictions),
                    train_recall = recall_score(t1_b_train, t1_train_predictions),
                    train_f1 = f1_score(t1_b_train, t1_train_predictions),
                    test_accuracy = accuracy_score(t1_b_test, t1_test_predictions),
                    test_precision = precision_score(t1_b_test, t1_test_predictions),
                    test_recall = recall_score(t1_b_test, t1_test_predictions),
                    test_f1 = f1_score(t1_b_test, t1_test_predictions),
                )
            )
        time = default_timer() - start
        print(f"    Total runtime: {time/60:.2f} minutes")
    results = pd.DataFrame(results)
    if full:
        return results
    else:
        return results.pivot_table(
            index = "column", 
            values = [
                "train_accuracy", "train_precision", "train_recall", "train_f1", 
                "test_accuracy", "test_precision", "test_recall", "test_f1"
            ]
        )
    
# Same but for MTL models, which take all labels at once
def get_folds_MTL(X, y, n_splits = 5, shuffle = True, random_state = 1234):
    """
    y should be a multidimensional array of labels (for multiclass classification)
    """
    kf = StratifiedKFold(
        n_splits = n_splits, 
        shuffle = shuffle, 
        random_state = random_state
    )
    
    # y_strat is the set of labels used for stratification in stratified sampling
    # computed as the logical or among all labels in task 2 (same as task 1)
    y_strat = np.logical_or.reduce(y.T)
    output = []
    for train_idx, val_idx in kf.split(X, y_strat):
        t = (X[train_idx].toarray(), y[train_idx])
        v = (X[val_idx].toarray(), y[val_idx])
        output.append((t,v))
    return output

#####################################################
# Modifications from Juan
#####################################################

def get_folds_MTL_juan(X, Y, n_splits = 5, shuffle = True, random_state = 1234):
    """
    Y should be a multidimensional array of labels (for multiclass classification)
    """
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    kf = StratifiedKFold(
        n_splits = n_splits, 
        shuffle = shuffle, 
        random_state = random_state
    )
    
    # y_strat is the set of labels used for stratification in stratified sampling
    # computed as the logical or among all labels in task 2 (same as task 1)
    y_or = np.logical_or.reduce(Y, axis = 1)
    output = []
    for train_idx, val_idx in kf.split(X, y_or):
        output.append(
            (
                X[train_idx], 
                Y[train_idx], 
                X[val_idx], 
                Y[val_idx]
            )
        )
    return output

def validate_MTL_juan(
    X, 
    Y, 
    classifier, 
    n_splits = 5, 
    shuffle = True, 
    random_state = 1234, 
    full = False,
    verbose = True,
    **train_parameters
):
    print("X", X.shape)
    print("Y", Y.shape)
    print("Cross-validation process started...")
    start = default_timer()
    results = []
    splits = get_folds_MTL_juan(
        X.values, Y.values, n_splits, shuffle, random_state
    )
    for i, (x_train, y_train, x_test, y_test) in enumerate(splits, 1):
        if verbose:
            print(f"*** fold {i} / {len(splits)}")
            print("    training model...")
        classifier.entrenar(
            x_train, 
            y_train, 
            x_test, 
            y_test, 
            **train_parameters
        )
        if verbose: 
            print("    generating predictions on the train set...")
        train_predictions = classifier.predecir(x_train) 
        if verbose: 
            print("    generating predictions on the test set...")
        test_predictions = classifier.predecir(x_test) 
        for j, col in enumerate(Y.columns):
            metrics = compute_metrics(
                y_train[:, j], 
                train_predictions[:, j], 
                y_test[:, j], 
                test_predictions[:, j]
            )
            results.append(
                dict(
                    fold = i,
                    column = col, 
                    **metrics
                )
            )
        # adding task_1 predictions
        t1_train_predictions = np.logical_or.reduce(train_predictions)
        t1_test_predictions = np.logical_or.reduce(test_predictions)
        t1_y_train = np.logical_or.reduce(y_train)
        t1_y_test = np.logical_or.reduce(y_test)
        metrics = compute_metrics(
            t1_y_train, 
            t1_train_predictions,
            t1_y_test,
            t1_test_predictions
        )
        results.append(
                dict(
                    fold = i,
                    column = 'task_1',
                    **metrics
                )
            )
        time = default_timer() - start
        print(f"    Total runtime: {time/60:.2f} minutes")
    results = pd.DataFrame(results)
    if full:
        return results
    else:
        return results.pivot_table(
            index = "column", 
            values = [
                "train_accuracy", "train_precision", "train_recall", "train_f1", 
                "test_accuracy", "test_precision", "test_recall", "test_f1"
            ]
        )

def compute_metrics(
    train_true, 
    train_predicted,
    test_true,
    test_predicted
):
    return dict(
        # train set
        train_accuracy = accuracy_score(train_true, train_predicted),
        train_precision = precision_score(
            train_true, 
            train_predicted, 
            zero_division = 0.0
        ),
        train_recall = recall_score(
            train_true, 
            train_predicted, 
            zero_division = 0.0
        ),
        train_f1 = f1_score(train_true, train_predicted, zero_division = 0.0),
        # test set
        test_accuracy = accuracy_score(test_true, test_predicted),
        test_precision = precision_score(
            test_true, 
            test_predicted, 
            zero_division = 0.0
        ),
        test_recall = recall_score(test_true, test_predicted, zero_division = 0.0),
        test_f1 = f1_score(test_true, test_predicted, zero_division = 0.0)
    )