import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split as split

def estimate_nbins(y):
    """
    Break down target vartiable into bins.

    Args:
        y (pd.Series): stratification target variable.

    Returns:
        bins (array): bins' values.

    """
    if len(y)/10 <= 100:
        nbins = int(len(y)/10)
    else:
        nbins = 100
    bins = np.linspace(min(y), max(y), nbins)
    return bins

def combine_single_valued_bins(y_binned):
    """
    Correct the assigned bins if some bins include a single value (can not be split).

    Find bins with single values and:
        - try to combine them to the nearest neighbors within these single bins
        - combine the ones that do not have neighbors among the single values with
        the rest of the bins.

    Args:
        y_binned (array): original y_binned values.

    Returns:
        y_binned (array): processed y_binned values.

    """
    # count number of records in each bin
    y_binned_count = dict(Counter(y_binned))
    # combine the single-valued-bins with nearest neighbors
    keys_with_single_value = []
    for key, value in y_binned_count.items():
        if value == 1:
            keys_with_single_value.append(key)

    # first combine with singles in keys_with_single_values
    def combine_singles(val, lst, operator, keys_with_single_value):
        for ix, v in enumerate(lst):
            if v == val:
                combine_with = lst[ix] + 1 if operator == 'subtract' else lst[ix] - 1
                y_binned[y_binned == val] = combine_with
                keys_with_single_value = [x for x in keys_with_single_value if x not in [val, combine_with]]
                y_binned_count[combine_with] = y_binned_count[combine_with] + 1 if operator == 'subtract' else y_binned_count[combine_with] - 1
                if val in y_binned_count.keys():
                    del y_binned_count[val]
        return keys_with_single_value
    for val in keys_with_single_value:
        # for each single value:
            # create lists based on keys_with_single_value with +-1 deviation
            # use these lists to find a match in keys_with_single_value
        lst_without_val = [i for i in keys_with_single_value if i != val]
        add_list = [x+1 for x in lst_without_val]
        subtract_list = [x-1 for x in lst_without_val]

        keys_with_single_value = combine_singles(val, subtract_list, 'subtract', keys_with_single_value)
        keys_with_single_value = combine_singles(val, add_list, 'add', keys_with_single_value)

    # now conbine the leftover keys_with_single_values with the rest of the bins
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for val in keys_with_single_value:
        nearest = find_nearest([x for x in y_binned if x not in keys_with_single_value], val)
        ix_to_change = np.where(y_binned == val)[0][0]
        y_binned[ix_to_change] = nearest

    return y_binned


def scsplit(*args, stratify, test_size = 0.3, train_size = 0.7, continuous = True, random_state = None):
    """
    Create stratfied splits for based on categoric or continuous column.

    For categoric target stratification raw sklearn is used, for continuous target
    stratification binning of the target variable is performed before split.

    Args:
        *args (pd.DataFrame/pd.Series): one dataframe to split into train, test
            or X, y to split into X_train, X_val, y_train, y_val.
        stratify (pd.Series): column used for stratification. Can be either a
        column inside dataset:
            train, test = scsplit(data, stratify = data['col'],...)
        or a separate pd.Series object:
            X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y).
        test_size (float): test split size. Defaults to 0.3.
        train_size (float): train split size. Defaults to 0.7.
        continuous (bool): continuous or categoric target variabale. Defaults to True.
        random_state (int): random state value. Defaults to None.

    Returns:
        if a single object is passed for stratification (E.g. 'data'):
            return:
                train (pd.DataFrame): train split
                valid (pd.DataFrame): valid split
        if two objects are passed for stratification (E.g. 'X', 'y'):
            return:
                X_train (pd.DataFrame): train split independent features
                X_val (pd.DataFrame): valid split independent features
                X_train (pd.DataFrame): train split target variable
                X_train (pd.DataFrame): valid split target variable

    """
    if random_state:
        np.random.seed(random_state)

    if len(args) == 2:
        X = args[0]
        y = args[1]
    else:
        X = args[0].drop(stratify.name, axis = 1)
        y = args[0][stratify.name]

    # non continuous stratified split (raw sklearn)
    if not continuous:
        y = np.array(y)
        y = combine_single_valued_bins(y)
        if len(args) == 2:
            X_train, X_val, y_train, y_val = split(X, y,
                                                   stratify = y,
                                                   test_size = test_size if test_size else None,
                                                   train_size = train_size if train_size else None)
            return X_train, X_val, y_train, y_val
        else:
            temp = pd.concat([X, pd.DataFrame(y, columns = [stratify.name])], axis= 1)
            train, val = split(temp,
                                stratify = temp[stratify.name],
                                test_size = test_size if test_size else None,
                                train_size = train_size if train_size else None)
            return train, val
    # ------------------------------------------------------------------------
    # assign continuous target values into bins
    bins = estimate_nbins(y)
    y_binned = np.digitize(y, bins)
    # correct bins if necessary
    y_binned = combine_single_valued_bins(y_binned)

    # split
    if len(args) == 2:
        X_t, X_v, y_t, y_v = split(X, y_binned,
                                   stratify = y_binned,
                                   test_size = test_size if test_size else None,
                                   train_size = train_size if train_size else None)

        try:
            X_train = X.iloc[X_t.index]
            y_train = y.iloc[X_t.index]
            X_val = X.iloc[X_v.index]
            y_val = y.iloc[X_v.index]
        except IndexError as e:
            raise Exception(f'{e}\nReset index of dataframe/Series before applying scsplit')
        return X_train, X_val, y_train, y_val
    else:
        temp = pd.concat([X, pd.DataFrame(y_binned, columns = [stratify.name])], axis= 1)
        tr, te = split(temp,
                       stratify = temp[stratify.name],
                       test_size = test_size if test_size else None,
                       train_size = train_size if train_size else None)
        train = args[0].iloc[tr.index]
        test = args[0].iloc[te.index]
        return train, test
