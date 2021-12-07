 # verstack - tools for applied machine learning

Machine learning tools to make a Data Scientist's work efficient

veratack package contains the following tools:

1. NaNImputer 
    *impute all missing values in a pandas dataframe using advanced machine learning with 1 line of code. Powered by XGBoost
2. Multicore 
    *execute any function in concurrency using all the available cpu cores
3. ThreshTuner
    *tune threshold for binary classification predictions
4. stratified_continuous_split 
    *create train/test splits stratified on the continuous variable
5. categoric_encoders encode categoric variable by numeric labels:
    * Factorizer - encode categoric variable by numeric labels
    * OneHotEncoder represent categoric variable as a set of binary variables
    * FrequencyEncoder - encode categoric variable by class frequencies
    * MeanTargetEncoder - encode categoric variable by mean of the target variable
    * WeightOfEvidenceEncoder - encode categoric variable as a weight of evidence of a binary target variable
 
6. timer 
    * convenient timer decorator to quickly measure and display time of any function execution


With NaNImputer you can fill missing values in numeric, binary and categoric columns in your pandas dataframe using advanced XGBRegressor/XGBClassifier models with just 1 line of code. Regardless of the data types in your dataframe (string/bool/numeric): 
 - all of them will be checked for missing values
 - transformed into numeric formats
 - split into the subsets with and without missing values
 - applicalbe models will be selected and configured for each of the columns with NaNs
 - models will be trained in multiprocessing mode utilizing all the available cores and threads (this saves a lot of time)
 - NaNs will be predicted and placed into corresponding indixes
 - columns with all NaNs will be droped
 - columns with NaNs and all other constants will be dropped
 - data will be reverse-transformed into original format

The module is highly configurable with default argumets set for the highest performance and verbosity

The only limitation is:
- NaNs in pure text columns are not imputed. By default they are filled with 'Missing_data' value. Configurable. If disabled - will return these columns with missing values untouched


## Usage

In the following paragraphs, I am going to describe how you can get and use verstack for your own projects.

###  Getting it

To download verstack, either fork this github repo or simply use Pypi via pip
```sh
$ pip install verstack
$ pip install --upgrade verstack
```

# NaNImputer
NaNImputer was programmed with ease-of-use in mind. First, import the NaNImputer class from verstack

```Python
from verstack import NaNImputer
```

And you are ready to go!

#### Initialize NaNImputer
First, let's create an imputer class instannce. We will not pass any argumets for this example and use all the defaults

```Python
imputer = NaNImputer()
```

#### Impute missing values in all columns of your dataframe
All you need to do is pass your dataframe as an only argument to the impute method of your imputer object and store the results in another object

```Python
df_without_nans = imputer.impute(df)
```
By default you will see all messages corresponding to each column imputation progress, data dimensions, utilized cores, timing

In most cases the resulting dataframe (df_without_nans according to our example) will have all the missing values imputed
For now missing values in text columns (not categorical) with over 500 unique values can not be imputed. By default they will be filled with 'Missing_data' string. This action can be disabled

#### Configuring NaNImputer
All the class configuration arguments are passed at class initialization
```Python 
# E.g.
imputer = NaNImputer(verbose = False)
```
##### Parameters

    conservative (bool, optional):
        Model complexity level used to impute missing values.
        If True: model will be set to less complex and much faster.
        Default = False
    n_feats (int, optional):
        Number of corellated independent features to be used for
        corresponding column (with NaN) model training and imputation.
        Default = 10
    nan_cols (list, optional):
        List of columns to impute missing values in.
        If None - all columns with missing values will be used.
        Default = None
    fix_string_nans (bool, optional):
        Find possible missing values in numeric columns that had been
        (mistakenly) encoded as strings, E.g. 'Missing'/'NaN'/'No data'
        and replace them with np.nan for further imputation
        Default = True
    multiprocessing_load (int, optional):
        Levels of parallel multiprocessing compute
        1 = single core
        2 = half of all available cores
        3 = all available cores
        Default = 3
    verbose (bool, optional):
        Print the imputation progress.
        Default = True
    fill_nans_in_pure_text (bool, optional):
        Fill the missing values in text fields by string 'Missing_data'.
        Applicable for text fields (not categoric).
        Default = True
    drop_empty_cols (bool, optional):
        Drop columns with all NaNs
        Default = True
    drop_nan_cols_with_constant (bool, optional):
        Drop columns containing NaNs and all other constant values
        Default = True
    feature_selection (str, optional):
        Define algorithm to select most important feats for each
        column imputation. Options: "correlation"/"feature_importance"
        Default = "correlation"        

##### Methods

    impute(data):
        Execute NaNs imputation columnwise in a pd.DataFrame

To impute missing values in a list of specific columns, use 20 most important features for each of these columns imputation and deploy a half of the available cpu cores, settings are as follows:
```Python 
imputer = NaNImputer(nan_cols = ['col1', 'col2'], n_feats = 20, multiprocessing_load = 2)
df_imputed = imputer.impute(df)
```

# Multicore

Execute any function in concurrency using all the available cpu cores.

```Python
from verstack import Multicore
```

#### Initialize Multicore
First, let's create a worker class instannce. We will not pass any argumets for this example and use all the defaults

```Python
worker = Multicore()
```

#### Use Multicore with default parameters

```Python
worker = Multicore()
result = worker.execute(function, iterable_list)
```
Execution time will be printed.

#### Limit number of workers and pass multiple iterables

```Python
worker = Multicore(workers = 2, multiple_iterables = True)
result = worker.execute(function, [iterable_dataframe, iterable_list])
```

##### Parameters

    workers (int or bool, optional):
        Number of workers if passed by user. If ``False``: all available cpu cores will be used.
        Default = False
    multiple_iterables (bool, optional):
        If function needs to iterate over multiple iterables, set to ``True``.
        Multiple iterables must be passed as a list (see examples below).
        Default = False

##### Methods

    Multicore.execute(func, iterable):
        Execute passed function and iterable(s) in concurrency.

##### Examples
  
Pass a function, one iterable object and use all the available cpu cores

```Python
  from verstack import Multicore

  worker = Multicore()
  result = worker.execute(function, iterable_list)
```

Pass a function, multiple iterable objects and use a limited number of cpu cores

```Python
  from verstack import Multicore

  worker = Multicore(workers = 2, multiple_iterables = True)
  result = worker.execute(function, [iterable_dataframe, iterable_list])
  # note that multiple iterable objects must be passes as a list to execute() method
```

# ThreshTuner
Find the best threshold to split your predictions in a binary classification task. Most applicable for imbalance target cases. 
In addition to thresholds & loss_func scores, the predicted_ratio (predicted fraction of 1) will be calculated and saved for every threshold. This will help the identify the appropriate threshold not only based on the score, but also based on the resulting distribution of 0 and 1 in the predictions.

```Python
from verstack import ThreshTuner
```

#### Initialize ThreshTuner
First, let's create an thresh class instannce. We will not pass any argumets for this example and use all the defaults

```Python
thresh = ThreshTuner()
```

#### Find the best threshold using default parameters
All you need to do is pass your labels and predictions as the only arguments to the fit method of your ThreshTuner object and the results will be stored in the class instance placeholders

```Python
thresh.fit(lables, pred)
```
By default 200 thresholds will be tested using the balanced_accuracy_score. The minimum and maximum thresholds will be inferred from the labels distribution (fraction_of_1)

#### Configuring ThreshTuner

```Python 
# E.g.
thresh = ThreshTuner(n_thresholds = 500)
```
##### Parameters

    n_thresholds (int, optional):
        Number of thresholds to test.
        Default = 200
    min_threshold (float/int, optional):
        Minimum threshold value. If not set by user: will be infered from labels balance based on fraction_of_1
        Default = None
    max_threshold (float/int, optional):
        Maximum threshold value. If not set by user: will be infered from labels balance based on fraction_of_1
        Default = None

##### Methods

    fit(labels, pred, loss_func):
        Calculate loss_func results for labels & preds for the defined/default thresholds. Print the threshold(s) with the best loss_func scores

        Parameters
        labels (array/list/series) [default=balanced_accuracy_score]
          y_true labels represented as 0 or 1

        pred (array/list/series)
          predicted probabilities of 1

        loss_func (function)
          loss function for scoring the predictions, e.g. sklearn.metrics.f1_score

    result():
        Display a dataframe with thresholds/loss_func_scores/fraction_of_1 for for all the the defined/default thresholds

    best_score():
        Display a dataframe with thresholds/loss_func_scores/fraction_of_1 for the best loss_func_score

    best_predict_ratio():
        Display a dataframe with thresholds/loss_func_scores/fraction_of_1 for the (predicted) fraction_of_1 which is closest to the (actual) labels_fraction_of_1

To configure ThreshTuner use the following logic:
```Python 
from sklearn.metrics import f1_score

thresh = ThreshTuner(n_thresholds = 500, min_threshold = 0.2, max_threshold = 0.6)
thresh.fit(labels, pred, f1_score)
```

To access the results:
```Python
thresh = ThreshTuner()
thresh.fit(labels, pred)

# return pd.DataFrame with all the results
thresh.result
# return pd.DataFrame with the best loss_func score
thresh.best_score()
thresh.best_score()['threshold']
# return pd.DataFrame with the best predicted fraction_of_1
thresh.best_predict_ratio()
# return the actual labels fraction_of_1
thresh.labels_fractio_of_1
```

# stratified_continuous_split

Create stratified splits based on either continuous or categoric target variable.
  - For continuous target variable verstack uses binning and categoric split based on bins
  - For categoric target enhanced sklearn.model_selection.train_test_split is used: in case there are not enough categories for the split, the minority classes will be combined with nearest neighbors.

Can accept only pandas.DataFrame/pandas.Series as data input.
```Python
  verstack.stratified_continuous_split.scsplit(*args, 
                                               stratify, 
                                               test_size = 0.3, 
                                               train_size = 0.7, 
                                               continuous = True, 
                                               random_state = None)
```

##### Parameters

    X,y,data: (pd.DataFrame/pd.Series)
        data input for the split in pandas.DataFrame/pandas.Series format.
    stratify (pd.Series): 
        target variable for the split in pandas/eries format.
    test_size (float, optional):
        test split ratio. Default = 0.3
    train_size (float, optional):
        train split ratio. Default = 0.3
    continuous (bool, optional):
        stratification target definition. If True, verstack will perform the stratification on the continuous target variable, if False, sklearn.model_selection.train_test_split will be performed with verstack enhancements. Default = True
    random_state (int, optional):
        random state value.
        Default = 5

##### Examples
  
```Python
  from verstack.stratified_continuous_split import scsplit
  
  train, test = scsplit(data, stratify = data['continuous_column_name'])
  X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y, 
                                           test_size = 0.3, random_state = 5)
```










# Factorizer

Encode categoric column by numeric labels.

Assign numeric labels starting with 0 to all unique variable's categories. 

Missing values can be encoded by an integer value (defaults to -1) / float / string or can be left untransformed.

When transform () - unseen categories will be be represented as NaN.

#### Initialize Factorizer

```Python
  from verstack import Factorizer
  # initialize with default parameters
  factorizer = Factorizer()
  # initialize with changing the NaN encoding value
  factorizer = Factorizer(na_sentinel = np.nan) #-999/0.33333/'No data')
```

##### Attributes

    na_sentinel
        Defined (at init) missing values encoding value. 
    colname
        Defined (at fit_transform()) column that had been transformed. 
    pattern
        Defined (at fit_transform()) encoding map.

##### Parameters

    na_sentinel [default=-1]
        Missing values encoding value. Can take int/float/str/np.nan values.

##### Methods

    fit_transform(df, colname)
        Fit Factorizer to data and return transformed data.
        Parameters
          df pd.DataFrame
              df containing the colname to transform.
          colname str
              Column name in df to be transformed.

    transform(df)
        Apply the fitted Factorizer to new data and return transformed data. Unseen categories will be represented by NaN.
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

    inverse_transform(df)
        Inverse transform data that had been encoded by Factorizer. Data must contain colname that was passed at fit_transform().
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

##### Examples

Use with default na_sentinel:

```Python

  factorizer = Factorizer()
  train_encoded = factorizer.fit_transform(train, 'colname') # will encode NaN values by -1
  test_encoded = factorizer.transform(test)

  train_reversed_to_original = factorizer.inverse_transform(train_encoded)
  test_reversed_to_original = factorizer.inverse_transform(test_encoded)

# keep missing values untransformed:

  factorizer = Factorizer(na_sentinel = np.nan)
  train_encoded = factorizer.fit_transform(train)
```




# OneHotEncoder

Encode categoric column by a set of binary columns.

Categoric 'column':['a','b','c'] will be represented by three binary columns 'a', 'b', 'c'. Original categoric 'column' is droped.
Missing values can be represented by a separate column or omited.
When transform() - unseen categories will not be represented by new columns, missing categories will be represented by empty (all zeros) columns.

#### Initialize OneHotEncoder

```Python
  from verstack import OneHotEncoder
  ohe = OneHotEncoder()
  train_encoded = ohe.fit_transform(train, 'colname') # will create a separate column for NaN values (if any)
  test_encoded = ohe.transform(test)

  train_reversed_to_original = ohe.inverse_transform(train_encoded)
  test_reversed_to_original = ohe.inverse_transform(test_encoded)
```

##### Attributes

    na_sentinel 
        Defined (at init) missing values encoding value. 
    colname
        Defined (at fit_transform()) column that had been transformed. 
    categories
        Defined (at fit_transform()) unique class categories which will be represented by binary columns.

##### Parameters

    na_sentinel [default=True]
        If True: create separate class column for NaN values.

##### Methods

    fit_transform(df, colname, prefix)
        Fit OneHotEncoder to data and return transformed data.
        Parameters
          df pd.DataFrame
              df containing the colname to transform.
          colname str
              Column name in df to be transformed.
          prefix str/int/float/bool/None, optional
              String to append DataFrame column names. The default is None.

    transform(df)
        Apply the fitted OneHotEncoder to new data and return transformed data. Unseen categories will not be represented by new columns, missing categories will be represented by empty (all zeros) columns.
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

    inverse_transform(df)
        Inverse transform data that had been encoded by OneHotEncoder. Data must contain one-hot-encoded columns that was created at fit_transform().
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

##### Examples

```Python
  ohe = OneHotEncoder()
  train_encoded = ohe.fit_transform(train, 'colname', prefix = 'colname')
  test_encoded = ohe.transform(test)

  train_reversed_to_original = ohe.inverse_transform(train_encoded)
  test_reversed_to_original = ohe.inverse_transform(test_encoded)
```




# FrequencyEncoder

Encoder to represent categoric variable classes' frequency across the dataset.

Original column ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'c', np.nan]
Encoded column  [0.3, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.1] # np.nan]

When transform() - unseen categories will be represented by the most common (highest) frequency.

Can handle missing values - encode NaN by NaN frequency or leave NaN values untransformed.
Resulting frequencies are normalized as a percentage.

#### Initialize FrequencyEncoder

```Python
  from verstack import FrequencyEncoder
  fe = FrequencyEncoder()
  train_encoded = fe.fit_transform(train, 'colname')
  test_encoded = fe.transform(test)

  train_reversed_to_original = fe.inverse_transform(train_encoded)
  test_reversed_to_original = fe.inverse_transform(test_encoded)
```

##### Attributes

    na_sentinel 
        Defined (at init) missing values encoding value. 
    colname
        Defined (at fit_transform()) column that had been transformed. 
    pattern
        Defined (at fit_transform()) encoding map.

##### Parameters

    na_sentinel [default=True]
        If True: Encode NaN values by their frequency. If False return np.nan in the encoded column.

##### Methods

    fit_transform(df, colname)
        Fit FrequencyEncoder to data and return transformed data.
        Parameters
          df pd.DataFrame
              df containing the colname to transform.
          colname str
              Column name in df to be transformed.

    transform(df)
        Apply the fitted FrequencyEncoder to new data and return transformed data. Unseen categories will be represented as NaN.
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

    inverse_transform(df)
        Inverse transform data that had been encoded by FrequencyEncoder. Data must contain colname that was passed at fit_transform().
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

##### Examples

```Python
  frequency_encoder = FrequencyEncoder()
  train_encoded = frequency_encoder.fit_transform(train, 'colname')
  test_encoded = frequency_encoder.transform(test)

  train_reversed_to_original = frequency_encoder.inverse_transform(train_encoded)
  test_reversed_to_original = frequency_encoder.inverse_transform(test_encoded)
```




# MeanTargetEncoder

Encode train cat cols by mean target value for category.

To avoid target leakage train set encoding is performed by breaking data into 5 folds & 
encoding categories of each fold with their respective target mean values calculated on the other 4 folds.
This will introduce minor noize to train data encoding (at fit_transform()) as a normalization technique. 
Test set (transform()) is encoded without normalization.

When transform() - unseen categories will be represented by the global target mean.

Can handle missing values - encode NaN by global mean or leave NaN values untransformed.

#### Initialize MeanTargetEncoder

```Python
  from verstack import MeanTargetEncoder
  mean_target_encoder = MeanTargetEncoder(save_inverse_transform = True)
  train_encoded = mean_target_encoder.fit_transform(train, 'colname', 'targetname')
  test_encoded = mean_target_encoder.transform(test)

  train_reversed_to_original = mean_target_encoder.inverse_transform(train_encoded)
  test_reversed_to_original = mean_target_encoder.inverse_transform(test_encoded)
```

##### Attributes

    na_sentinel
        Defined (at init) missing values encoding value. 
    colname
        Defined (at fit_transform()) column that had been transformed. 
    pattern
        Defined (at fit_transform()) encoding map.
    save_inverse_transform
        Defined (at init) flag for saving the pattern for inverse transform.

##### Parameters

    na_sentinel [default=True]
        If True: Encode NaN values by target global mean. If False return np.nan in the encoded column.
    save_inverse_transform [default=False]
        If True: Saves mean target values for each category at each encoding fold. Enable if need to inverse_transform the encoded data. Defaults to False because for large datasets saved pattern can significantly increase instance object size.

##### Methods

    fit_transform(df, colname, targetname)
        Fit MeanTargetEncoder to data and return transformed data.
        Parameters
          df pd.DataFrame
              df containing the colname to transform.
          colname str
              Column name in df to be transformed.
          targetname str
              Target column name in df for extracting the mean values for each colname category.

    transform(df)
        Apply the fitted MeanTargetEncoder to new data and return transformed data. Unseen categories will be encoded by the global target mean.
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

    inverse_transform(df)
        Inverse transform data that had been encoded by MeanTargetEncoder. Data must contain colname that was passed at fit_transform().
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

##### Examples

```Python
  mean_target_encoder = MeanTargetEncoder(save_inverse_transform = True)
  train_encoded = mean_target_encoder.fit_transform(train, 'colname', 'targetname')
  test_encoded = mean_target_encoder.transform(test)

  train_reversed_to_original = mean_target_encoder.inverse_transform(train_encoded)
  test_reversed_to_original = mean_target_encoder.inverse_transform(test_encoded)
```




# WeightOfEvidenceEncoder

Encoder to represent categoric variables by Weight of Evidence in regards to the binary target variable.

Built on top of sclearn package [category_encoders.woe.WOEEncoder](https://contrib.scikit-learn.org/category_encoders/woe.html#).

If encoded value is negative - it represents a category that is more heavily enclided to the negative target class (0).
Positive encoding result represents inclination to the positive target class (1).

When fit_transform() is used on a train set, variable is encoded with adding minor noize to reduce the risk of overfitting.
Can handle missing values - encode NaN by zero WoE or leave NaN untransformed.

#### Initialize WeightOfEvidenceEncoder

```Python
  from verstack import WeightOfEvidenceEncoder
  WOE = WeightOfEvidenceEncoder()
  train_encoded = WOE.fit_transform(train, 'colname', 'targetname')
  test_encoded = WOE.transform(test)

  train_reversed_to_original = WOE.inverse_transform(train_encoded)
  test_reversed_to_original = WOE.inverse_transform(test_encoded)
```

##### Attributes

    na_sentinel 
        Defined (at init) missing values encoding value. 
    colname
        Defined (at fit_transform()) column that had been transformed. 
    params
        Defined (at init) category_encoders.woe.WOEEncoder `parameters <https://contrib.scikit-learn.org/category_encoders/woe.html#>`_


##### Parameters

    na_sentinel [default=True]
        If True: Encode NaN values by zero WoE. If False return np.nan in the encoded column.
    kwargs
       category_encoders.woe.WOEEncoder [parameters](https://contrib.scikit-learn.org/category_encoders/woe.html#). Following parameters are set by default: ``'randomized':True``, ``'random_state':42``, ``'handle_missing':'return_nan'`` <- infered from na_sentinel setting.

##### Methods

    fit_transform(df, colname, targetname)
        Fit WeightOfEvidenceEncoder to data and return transformed data.
        Parameters
          df pd.DataFrame
              df containing the colname to transform.
          colname str
              Column name in df to be transformed.
          targetname str
              Target column name in df for calculating WoE for each colname category.

    transform(df)
        Apply the fitted WeightOfEvidenceEncoder to new data and return transformed data. Unseen categories' WoE is set to 0.
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

    inverse_transform(df)
        Inverse transform data that had been encoded by WeightOfEvidenceEncoder. Data must contain colname that was passed at fit_transform().
        Parameters
          df pd.DataFrame
              Data containing the colname to transform.

##### Examples

```Python
  WOE = WeightOfEvidenceEncoder()
  train_encoded = WOE.fit_transform(train, 'colname', 'targetname')
  test_encoded = WOE.transform(test)

  train_reversed_to_original = WOE.inverse_transform(train_encoded)
  test_reversed_to_original = WOE.inverse_transform(test_encoded)
```










# timer

timer is a decorator function: it must placed above the function (that needs to be timed) definition.

```Python
  verstack.tools.timer
```

##### Examples
  
```Python
  from verstack.tools import timer

  @timer
  def func(a,b):
      print(f'Result is: {a + b}')

  func(2,3)

  >>>Result is: 5
  >>>Time elapsed for func execution: 0.0002 seconds
```

Experiment with different settings for your application, and if anything does not work as expected, feel free to reach out to me at danil.com@me.com

License
----

MIT License

Copyright (c) 2020 DanilZherebtsov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

danil.com@me.com
