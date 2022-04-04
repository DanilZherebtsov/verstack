verstack 3.0.0 Documentation
============================

Machine learning tools to make a Data Scientist\'s work efficient

veratack package contains the following tools:

-   **Stacker** automated stacking ensemble configuration/train/features
    creation in train/test sets
-   **DateParser** automated date columns finder and parser
-   **LGBMTuner** automated lightgbm models tuner with optuna
-   **NaNImputer** impute all missing values in a pandas dataframe using
    advanced machine learning with 1 line of code
-   **Multicore** execute any function in concurrency using all the
    available cpu cores
-   **ThreshTuner** tune threshold for binary classification predictions
-   **stratified\_continuous\_split** create train/test splits
    stratified on the continuous variable
-   **categoric\_encoders** encode categoric variable by numeric labels

> -   **Factorizer** encode categoric variable by numeric labels
> -   **OneHotEncoder** represent categoric variable as a set of binary
>     variables
> -   **FrequencyEncoder** encode categoric variable by class
>     frequencies
> -   **MeanTargetEncoder** encode categoric variable by mean of the
>     target variable
> -   **WeightOfEvidenceEncoder** encode categoric variable as a weight
>     of evidence of a binary target variable

-   **timer** convenient timer decorator to quickly measure and display
    time of any function execution
-   **pretty_print** a convenient function to set up and execute print statements based on the 'global' verbosity setting within large projects

Getting verstack

\$ `pip install verstack`

\$ `pip install --upgrade verstack`


Stacker
-------

Fully automated highly configurable stacking ensemble creation class.
Can create single or multiple layers of stacked features. Applicable for
train/test set features creation. Any number of layers and models within
layers can be added to Stacker instance (models in layers must contain
fit / predict / predict\_proba [if classification]{.title-ref} methods
for the Stacker to properly create features using these models).

Additional metafeatures can be created from stacked features if
metafeats parameter is set to True.

Subsequent (\>1) layers can be trained either on predictions from one
previous layer / or predictions from one previous layer and meta
features / or predictions from all previous layers / or predictions from
all previous layers and meta features [subject to stacking\_feats\_depth
parameter configuration]{.title-ref}; original X features can also be
used for training the subsequent layers [subjuect to include\_X
parameter configuration]{.title-ref}.

Stacker includes auto mode which will create two layers of stacked
features with layer 1 consisting of 14 diverse models and layer 2
consisting of a linear and boosed model

Models\' `RandomizedSearchCV` hyperparameters tuning is enabled if
gridsearch\_iteration parameter is \> 0 [subject to model being
supported by built in parameters optimization function]{.title-ref}.

Stacked feats creation on the train set is perfromed by train/predict
operations on 4 folds. Each stacked feature in the test set is created
by predicting with 4 models [fitted on train set]{.title-ref} and
averaging predictions. When averaging for regression tasks - mean of
predicted values is computed; for binary - mean of positive class
probabilities is computed; for multiclass - the most commonly predicted
class from the 4 predictions is selected.

> \... the output of fit\_transfrom() / transform() methods will return
> the dataframe with original features and stacked features.

**auto mode models**

> layer\_1: 14 models
>
> -   LGBM(max\_depth = 12)
> -   XGB(max\_depth = 10, n\_jobs = -1)
> -   GradientBoosting(max\_depth = 7)
> -   kerasModel(num\_layers = 3)
> -   kerasModel(num\_layers = 2)
> -   kerasModel(num\_layers = 1)
> -   ExtraTree(max\_depth = 12)
> -   RandomForest(max\_depth = 7)
> -   Linear/LogisticRegression()
> -   KNeighbors(n\_neighbors=15)
> -   KNeighbors(n\_neighbors=10)
> -   SVR(kernel = \'rbf\')
> -   DecisionTree(max\_depth = 15)
> -   DecisionTree(max\_depth = 8)
>
> layer\_2: two models
>
> -   LGBM(max\_depth = 3)
> -   Ridge()

**Initialize Stacker**

``` {.python}
from verstack import Stacker

# initialize with default parameters
stacker = Stacker(objective = 'regression')

# initialize with selected parameters
stacker = Stacker(objective = 'regression',
                  auto = True,
                  auto_num_layers = 2,
                  metafeats = True,
                  epochs = 500,
                  gridsearch_iterations = 20,
                  stacking_feats_depth = 1,
                  include_X = False,
                  verbose = True)
```

### Parameters

> parameters `metafeats`, `gridsearch_iterations`, `stacking_feats_depth`, `include_X` can be configured independently for any layer in the follwoing manner: E.g. If need to optimize the models\' hyperparameters only in layer\_2:
>
> :   -   `stacker = Stacker('regression', gridsearch_iterations = 0)`
>     -   `stacker.add_layer([model_1, model_2, model_3])`
>     -   `X_transformed = stacker.fit_transform(X, y)`
>     -   `stacker.add_layer([model_4, model_5])`
>     -   `stacker.gridsearch_iterations = 20`
>     -   `X_transformed = stacker.fit_transform(X_transformed, y)`

-   `objective` \[default=None\]

    Training objective. Can take values: \'regression\', \'binary\',
    \'multiclass\'

-   `auto` \[default=False\]

    Enable/disable automatic configuration of 1 or 2 layers of models to
    create stacked features. If True will automatically populate the
    self.layers with 1 or 2 lists of preconfigured diverse models.

-   `auto_num_layers` \[default=2\]

    Number of automatically generated layers. Can take values 1 and 2

-   `metafeats` \[default=True\]

    Additional statistical meta features creation from the stacked predictions:

    :   -   pairwise differences between the stacked predictions are
            created for all pairs (recursively)
        -   mean and std for all the stacked features in a layer are
            created as two extra meta feats

-   `epochs` \[default=200\]

    Number of neural networks epochs. Applicable for the three
    automatically configured neural networks in the auto mode

-   `gridsearch_iterations` \[default=10\]

    Number of hyperparameters optimization iterations. If set to 0,
    hyperparameters will not be optimized. If \> 0, hyperparameters in
    all layers will be optimized. E.g. Supported models for
    optimization:

    > -   lightgbm.sklearn.LGBMRegressor /
    >     lightgbm.sklearn.LGBMClassifier
    > -   xgboost.sklearn.XGBRegressor / xgboost.sklearn.XGBClassifier
    > -   sklearn.ensemble.GradientBoostingRegressor /
    >     sklearn.ensemble.GradientBoostingClassifier
    > -   sklearn.tree.\_classes.ExtraTreeRegressor /
    >     sklearn.tree.\_classes.ExtraTreeClassifier
    > -   sklearn.ensemble.\_forest.RandomForestRegressor /
    >     sklearn.ensemble.\_forest.RandomForestClassifier
    > -   sklearn.linear\_model.\_logistic.LogisticRegression
    > -   sklearn.linear\_model.\_ridge.Ridge
    > -   sklearn.neighbors.\_regression.KNeighborsRegressor /
    >     sklearn.neighbors.\_classification.KNeighborsClassifier
    > -   sklearn.svm.\_classes.SVR / sklearn.svm.\_classes.SVC
    > -   sklearn.tree.\_classes.DecisionTreeRegressor /
    >     sklearn.tree.\_classes.DecisionTreeClassifier

-   `stacking_feats_depth` \[default=1\]

    Defines the features used by subsequent (\>1) layers to train the stacking models. Can take values between 1 and 4 where:

    :   -   1 = use predictions from one previous layer
        -   2 = use predictions from one previous layer and meta
            features
        -   3 = use predictions from all previous layers
        -   4 = use predictions from all previous layers and meta
            features

-   `include_X` \[default=False\]

    Flag to use original X features for subsequent layer training

-   `verbose` \[default=True\]

    Print progress outputs or silent

### Methods

-   `add_layer([model_1, model_2(), model_3])`

    Add layer with models to Stacker instance.

    > Parameters
    >
    > -   `models_list` \[list\]
    >
    >     List containing initiated models instances. Each model must
    >     contain fit() / predict() / predict\_proba() [if
    >     classification]{.title-ref} methods

    returns

       None

-   `fit_transform(X, y)`

    Train/predict/append to X the stacking features from models defined
    in self.layers

    > Parameters
    >
    > -   `X` \[pd.DataFrame\]
    >
    >     train features
    >
    > -   `y` \[pd.Series\]
    >
    >     train labels

    returns

       pd.DataFrame train featues with appended stacking features

-   `transform(X)`

    Create stacking features on the test set from models saved in
    self.trained\_models

    > Parameters
    >
    > -   `X` \[pd.DataFrame\]
    >
    >     test features

    returns

       pd.DataFrame test featues with appended stacking features

**Attributes**

-   `layers`

    Dictionary with \'layer\_n\' as key and list of models in layer as
    value

-   `trained_models`

    Dictionary with \'layer\_n\' as key and dictionary with stacked
    feature name as key and list of 4 [trained on different
    folds]{.title-ref} models instances for predicting on test set

### Examples

Using Stacker in auto mode

``` {.python}
from verstack import Stacker
stacker = Stacker(objective = 'multiclass', auto = True)
X_with_stacked_feats = stacker.fit_transform(X, y)
```

Add two custom layers, for training subsequent (\>1) layers use not only
the predictions of the previous layer, but also metafeats in the
previous layer and X original features Then add one more layer and
disable hyperparameters optimization for this layer

``` {.python}
# initialize Stacker
stacker = Stacker(objective = 'multiclass', 
                  auto = False,
                  stacking_feats_depth = 2,
                  include_X = True)
# add layers
stacker.add_layer([model_1, model_2, model_3])
stacker.add_layer([model_4, model_5])
# add stacking features to train/test
X_with_stacked_feats = stacker.fit_transform(X, y)
test_with_stacked_feats = stacker.transform(test)
# add extra layer
stacker.add_layer([model_6, model_7])
# change the gridsearch_iteration setting
stacker.gridsearch_iterations = 0
# pass the transformed dataset if need to call .fit_transform() after adding extra layers to the fitted instance of Stacker
X_with_stacked_feats = stacker.fit_transform(X_with_stacked_feats, y)
test_with_stacked_feats = stacker.transform(test_with_stacked_feats)
```

DateParser
----------

Fully automated DateParser tool that takes as input a pandas.DataFrame
and returns a pandas.DataFrame with parsed datetime features. Holidays
flags and names are created as features subject to user passing the
country argument (E.g. country = \'US\'). Holiday features extraction
are based on utilizing the [holidays]{.title-ref} package. Datetime
columns will be found automatically, transformed to pd.Timestamp format,
new columns with the follwing features (if applicable to the specific
datetime format) will be created: - year - month - day (monthday) -
quarter - week - weekday - dayofyear - hour - minute - second -
part\_of\_day - timediff (if two datetime columns are found) -
is\_holiday (if country argument is passed) - holiday\_name (if country
argument is passed) - is\_payday (if payday argument is passed) -
days\_from\_epoch (1970/01/01)

> \... same set of features will be created (with column name prefix)
> for each of the datetime columns DateParser detects.

**Supported datetime formats**

> -   \'28-OCT-90\',
> -   \'28-OCT-1990\',
> -   \'10/28/90\',
> -   \'10/28/1990\',
> -   \'28.10.90\',
> -   \'28.10.1990\',
> -   \'90/10/28\',
> -   \'1990/10/28\',
> -   \'4 Q 90\',
> -   \'4 Q 1990\',
> -   \'OCT 90\',
> -   \'OCT 1990\',
> -   \'43 WK 90\',
> -   \'43 WK 1990\',
> -   \'01:02\',
> -   \'02:34\',
> -   \'02:34.75\',
> -   \'20-JUN-1990 08:03\',
> -   \'20-JUN-1990 08:03:00\',
> -   \'1990-06-20 08:03\',
> -   \'1990-06-20 08:03:00.0\'

**Initialize DateParser**

``` {.python}
from verstack import DateParser

# initialize with default parameters
parser = DateParser()

# initialize with selected parameters
parser = DateParser(country = 'US', 
                  state = 'CA',
                  payday = [1, 15])
```

### Parameters

-   `country` \[default=None\]

    Country name or abreviation. For a full list of supported countries
    call parser.list\_supported\_countries()

-   `state` \[default=None\]

    State abreviation. Correct state abreviations are available at
    <https://pypi.org/project/holidays/>

-   `prov` \[default=None\]

    Province abreviation. Correct province abreviations are available at
    <https://pypi.org/project/holidays/>

-   `payday` \[default=None\]

    List of paydays applicable in a specific country. E.g. \[1, 15\]

-   `verbose` \[default=True\]

    Enable or desable console prints

### Methods

-   `fit_transform(df)`

    Fully automatic search of datetime columns and features extraction.
    Apart from all the conventional datetime features will automatically
    parse holidays / paydays if specified and init. Saves the found
    datetime columns names and feature extraction pipelines for the
    transform() method.

    > Parameters
    >
    > -   `df` \[pd.DataFrame\]
    >
    >     Data with raw features

    returns

       pd.DataFrame with new features

-   `transform(df)`

    Parse identical set of features from a new dataset. Usually applied
    to test set transformation. E.g. if test set datetime columns
    include a short timeframe so that quarter feature is constant and
    thus should not be created, the dataset will still be populated by
    this feature in order to preserve the identical columns names and
    order between train/test sets. Think machine learning.

    > Parameters
    >
    > -   `df` \[pd.DataFrame\]
    >
    >     Data with raw features (test/valid set)

    returns

       pd.DataFrame with new features

-   `parse_holidays(datetime_col_series, country, state, province, holiday_names)`

    Create series with holidays names or flags for a defined country
    based on series of datetime-like strings.

    > -   `datetime_col_series` \[pd.Series\]
    >
    >     Series of datetime-like strings in line with
    >     supported\_formats
    >
    > -   `country` \[str\]
    >
    >     Country name or abreviation. For a full list of supported
    >     countries call parser.list\_supported\_countries()
    >
    > -   `state` \[str, default = None\]
    >
    >     State abreviation. Correct state abreviations are available at
    >     <https://pypi.org/project/holidays/>
    >
    > -   `prov` \[str, default = None\]
    >
    >     Province abreviation. Correct province abreviations are
    >     available at <https://pypi.org/project/holidays/>
    >
    > -   `holiday_names` \[bool, default = False\]
    >
    >     Flag to return holidays as a binary feature or string holidays
    >     names

    returns

       pd.Series with holidays binary flags or holidays string names

-   `get_holidays_calendar(country, years, state = None, prov = None)`

    Get data on the holidays in a given country (optinally in a certain
    state/province) for a given year(s).

    > -   `country` \[str\]
    >
    >     Country name or abreviation. For a full list of supported
    >     countries call parser.list\_supported\_countries()
    >
    > -   `state` \[str, default = None\]
    >
    >     State abreviation. Correct state abreviations are available at
    >     <https://pypi.org/project/holidays/>
    >
    > -   `prov` \[str, default = None\]
    >
    >     Province abreviation. Correct province abreviations are
    >     available at <https://pypi.org/project/holidays/>

    returns

       dictionary with holidays dates and names

-   `list_supported_countries()`

    Print a list of supported countries and abreviations.

**Attributes**

-   `datetime_cols`

    List of found datetime columns names. Available after
    fit\_transform()

-   `created_datetime_cols`

    List of created datetime features. Available after fit\_transform()

-   `supported formats`

    List of supported datetime formats

### Examples

Using DateParser with all default parameters

``` {.python}
parser = DateParser()
train_with_parsed_dt_feats = parser.fit_transform(train)
test_with_parsed_dt_feats = parser.transform(test)
```

DateParser with holidays/paydays

``` {.python}
parser = DateParser(country = 'US', payday = [1, 15])
train_with_parsed_dt_feats = parser.fit_transform(train)
test_with_parsed_dt_feats = parser.transform(test)
```

LGBMTuner
---------

Fully automated lightgbm model hyperparameter tuning class with optuna
under the hood. LGBMTuner selects optimal hyperparameters based on
executed trials (configurable), optimizes n\_estimators and fits the
final model to the whole train set. Feature importances are available in
numeric format, as a static plot, and as an interactive plot (html).
Optimization history and parameters importance in static and interactive
formats are alse accesable by built in methods.

### Logic

The only required user inputs are the X (features), y (labels) and
evaluation metric name, LGBMTuner will handle the rest

> -   lgbm model type (regression/classification) is inferred from the
>     labels and evaluation metric (passed by user)
> -   optimization metric may be different from the evaluation metric
>     (passed by user). LGBMTuner at hyperparameters search stage
>     imploys the error reduction strategy, thus:
>     -   most regression task type metrics are supported for
>         optimization, if not, MSE is selected for optimization
>     -   for classification task types hyperparameters are tuned by
>         optimizing log\_loss, n\_estimators are tuned with
>         evaluation\_metric
> -   early stopping is engaged at each stage of LGBMTuner optimizations
> -   for every trial (iteration) a random train\_test\_split is
>     performed (stratified for classification)
> -   lgbm model initial parameters!=defaults and are inferred from the
>     data stats and built in logic
> -   optimization parameters and their search space are inferred from
>     the data stats and built in logic
> -   LGBMTuner class instance (after optimization) can be used for
>     making predictions with conventional syntaxis
>     (predict/predict\_proba)
> -   verbosity is controlled and by default outputs only the necessary
>     optimization process/results information

**Initialize LGBMTuner**

``` {.python}
from verstack import LGBMTuner

# initialize with default parameters
tuner = LGBMTuner('rmse')

# initialize with selected parameters
tuner = LGBMTuner(metric = 'rmse', 
                  trials = 200, 
                  refit = False, 
                  verbosity = 0, 
                  visualization = False, 
                  seed = 999)
```

### Parameters

-   `metric` \[default=None\]

    Evaluation metric for hyperparameters optimization. LGBMTuner supports the following metrics (note the syntax)

    :   \[\'mae\', \'mse\', \'rmse\', \'rmsle\', \'mape\', \'smape\',
        \'rmspe\', \'r2\', \'auc\', \'gini\', \'log\_loss\',
        \'accuracy\', \'balanced\_accuracy\', \'precision\',
        \'precision\_weighted\', \'precision\_macro\', \'recall\',
        \'recall\_weighted\', \'recall\_macro\', \'f1\',
        \'f1\_weighted\', \'f1\_macro\', \'lift\'\]

-   `trials` \[default=100\]

    Number of trials to run

-   `refit` \[default=True\]

    Fit the model with optimized hyperparameters on the whole train set
    (required for feature\_importances, plot\_importances() and
    prediction methods)

-   `verbosity` \[default=1\]

    Console verbosity level: 0 - no output except for optuna CRITICAL
    errors and builtin exceptions; (1-5) based on optuna.logging
    options. The default is 1

-   `visualization` \[default=True\]

    Automatically output feature\_importance & optimization plots into
    the console after tuning. Plots are also available on demand by
    corresponding methods

-   `seed` \[default=42\]

    Random state parameter

### Methods

-   `fit(X, y)`

    Execute LGBM model hyperparameters tuning

    > Parameters
    >
    > -   `X` \[pd.DataFrame\]
    >
    >     Train features
    >
    > -   `y` \[pd.Series\]
    >
    >     Train labels

-   `optimize_n_estimators(X, y, params, verbose_eval = 100)`

    Optimize n\_estimators for lgb model.

    > Parameters
    >
    > -   `X` \[np.array\]
    >
    >     Train features
    >
    > -   `y` \[np.array\]
    >
    >     Train labels
    >
    > -   `params` \[dict\]
    >
    >     parameters to use for training the model with early stopping
    >
    > -   `verbose_eval` \[int\]
    >
    >     evaluation output at each `verbose_eval` iteratio n
    >
    > returns
    >
    > :   (best\_iteration, best\_score)

-   `fit_optimized(X, y)`

    Train model with tuned params on whole train data

    > -   `X` \[np.array\]
    >
    >     Train features
    >
    > -   `y` \[np.array\]

-   `predict(test, threshold = 0.5)`

    Predict by optimized model on new data

    > -   `test` \[pd.DataFrame\]
    >
    >     Test features
    >
    > -   `threshold` \[default=0.5\]
    >
    >     Classification threshold (applicable for binary
    >     classification)

    returns

       array of int

-   `predict_proba(test)`

    Predict probabilities by optimized model on new data

    > -   `test` \[pd.DataFrame\]
    >
    >     Test features

    returns

       array of float

-   `plot_importances(n_features = 15, figsize = (10,6), interactive = False)`

    Plot feature importance

    > -   `n_features` \[default=15\]
    >
    >     Number of important features to plot
    >
    > -   `figsize` \[default=(10,6)\]
    >
    >     plot size
    >
    > -   `interactive` \[default=False\]
    >
    >     Create & display with the default browser the interactive html
    >     plot or (if browser disply is unavailable) save to current wd.

-   `plot_optimization_history(interactive = False)`

    Plot optimization function improvement history

    > -   `interactive` \[default=False\]
    >
    >     Create & display with the default browser the interactive html
    >     plot or (if browser disply is unavailable) save to current wd.

-   `plot_param_importances(interactive = False)`

    Plot params importance plot

    > -   `interactive` \[default=False\]
    >
    >     Create & display with the default browser the interactive html
    >     plot or (if browser disply is unavailable) save to current wd.

-   `plot_intermediate_values(interactive = False, legend = False)`

    Plot optimization trials history. Shows successful and terminated
    trials. If trials \> 50 it is better to study the interactive
    version

    > -   `interactive` \[default=False\]
    >
    >     Create & display with the default browser the interactive html
    >     plot or (if browser disply is unavailable) save to current wd.
    >
    > -   `legend` \[default=False\]
    >
    >     Plot legen on a static plot

**Attributes**

-   `metric`

    Evaluation metric defined by user at LGBMTuner init

-   `refit`

    Setting for refitting the optimized model on whole train dataset

-   `verbosity`

    Verbosity level settings

-   `visualization`

    Automatic plots output after optimization setting

-   `seed`

    Random state value

-   `fitted_model`

    Trained LGBM booster model with optimized parameters

-   `feature_importances`

    Feature importance values

-   `study`

    optuna.study.study.Study object after hyperparameters tuning

-   `init_params`

    initial LGBM model parameters

-   `best_params`

    learned optimized parameters

### Examples

Using LGBMTuner with all default parameters

``` {.python}
imputer = LGBMTuner('auc')
tuner.fit(X, y)
tuner.feature_importances
tuner.plot_importances()
tuner.plot_intermediate_values()
tuner.plot_optimization_history()
tuner.plot_param_importances()
tuner.best_params
tuner.predict(test)
```

LGBMTuner with custom settings

``` {.python}
imputer = LGBMTuner(metric = 'auc', trials = 300, verbosity = 3, visualization = False)
tuner.fit(X, y)
tuner.plot_importances(legend = True)
tuner.plot_intermediate_values(interactive = True)
tuner.predict(test, threshold = 0.3)
```

NaNImputer
----------

Impute all missing values in a pandas dataframe by xgboost models in
multiprocessing mode using a single line of code.

### Logic

With NaNImputer you can fill missing values in numeric, binary and
categoric columns in your pandas dataframe using advanced
XGBRegressor/XGBClassifier models with just 1 line of code. Regardless
of the data types in your dataframe (string/bool/numeric):

> -   all of the columns will be checked for missing values
> -   transformed into numeric formats
> -   split into subsets with and without missing values
> -   applicalbe models will be selected and configured for each of the
>     columns with NaNs
> -   models will be trained in multiprocessing mode utilizing all the
>     available cores and threads of your cpu (this saves a lot of time)
> -   NaNs will be predicted and placed into corresponding indixes
> -   columns with all NaNs will be droped
> -   columns containing NaNs and known values as a single constant
> -   data will be reverse-transformed into original format

The module is highly configurable with default argumets set for the
highest performance and verbosity

The only limitation is:

-   NaNs in pure text columns are not imputed. By default they are
    filled with \'Missing\_data\' value. Configurable. If disabled -
    will return these columns with missing values untouched

**Initialize NaNImputer**

``` {.python}
from verstack import NaNImputer

# initialize with default parameters
imputer = NaNImputer()

# initialize with selected parameters
imputer = NaNImputer(conservative = False, 
                     n_feats = 10, 
                     nan_cols = None, 
                     fix_string_nans = True, 
                     multiprocessing_load = 3, 
                     verbose = True, 
                     fill_nans_in_pure_text = True, 
                     drop_empty_cols = True, 
                     drop_nan_cols_with_constant = True)
```

### Parameters

-   `conservative` \[default=False\]

    Model complexity level used to impute missing values. If `True`:
    model will be set to less complex and much faster.

-   `n_feats` \[default=10\]

    Number of corellated independent features to be used
    forcorresponding column (with NaN) model training and imputation.

-   `nan_cols` \[default=None\]

    List of columns to impute missing values in. If None: all the
    columns with missing values will be used.

-   `fix_string_nans` \[default=True\]

    Find possible missing values in numeric columns that had been
    (mistakenly) encoded as strings, E.g. \'Missing\'/\'NaN\'/\'No
    data\' and replace them with np.nan for further imputation.

-   `multiprocessing_load` \[default=3\]

    -   Levels of parallel multiprocessing compute
        -   1 = single core
        -   2 = half of all available cores
        -   3 = all available cores

-   `verbose` \[default=True\]

    Print the imputation progress.

-   `fill_nans_in_pure_text` \[default=True\]

    Fill the missing values in text fields by string
    \'Missing\_data\'.Applicable for text fields (not categoric).

-   `drop_empty_cols` \[default=True\]

    Drop columns with all NaNs.

-   `drop_nan_cols_with_constant` \[default=True\]

    Drop columns containing NaNs and known values as a single constant.

-   `feature_selection` \[default=\"correlation\"\]

    -   Define algorithm to select most important feats for each column
        imputation. Quick option: \"correlation\" is based on selecting
        n\_feats with the highest binary correlation with each column
        for NaNs imputation. Less quick but more precise:
        \"feature\_importance\" is based on extracting
        feature\_importances from an xgboost model.

### Methods

-   `impute(data)`

    Execute NaNs imputation columnwise in a pd.DataFrame

    > Parameters
    >
    > -   `data` pd.DataFrame
    >
    >     dataframe with missing values in a single/multiple columns

### Examples

Using NaNImputer with all default parameters

``` {.python}
imputer = NaNImputer()
df_imputed = imputer.impute(df)
```

Say you would like to impute missing values in a list of specific
columns, use 20 most important features for each of these columns
imputation and deploy a half of the available cpu cores

``` {.python}
imputer = NaNImputer(nan_cols = ['col1', 'col2'], n_feats = 20, multiprocessing_load = 2)
df_imputed = imputer.impute(df)
```

Multicore
---------

Execute any function in concurrency using all the available cpu cores.

### Logic

> Multicore module is built on top of concurrent.futures package. Passed
> iterables are divided into chunks according to the number of workers
> and passed into separate processes.
>
> Results are extracted from finished processes and combined into a
> single/multiple output as per the defined function output
> requirements.
>
> Multiple outputs are returned as a nested list.

**Initialize Multicore**

``` {.python}
from verstack import Multicore

# initialize with default parameters
multicore = Multicore()

# initialize with selected parameters
multicore = Multicore(workers = 6,
                      multiple_iterables = True)
```

### Parameters

-   `workers` int or bool \[default=False\]

    Number of workers if passed by user. If `False`: all available cpu
    cores will be used.

-   `multiple_iterables` bool \[default=False\]

    If function needs to iterate over multiple iterables, set to `True`.

    Multiple iterables must be passed as a list (see examples below).

### Methods

-   `execute(func, iterable)`

    Execute passed function and iterable(s) in concurrency.

    > Parameters
    >
    > -   `func` function
    >
    >     function to execute in parallel
    >
    > -   `iterable` list/pd.Series/pd.DataFrame/dictionary
    >
    >     data to iterate over

### Examples

Use Multicore with all default parameters

``` {.python}
multicore = Multicore()
result = multicore.execute(function, iterable_list)
```

If you want to use a limited number of cpu cores and need to iterate
over two objects:

``` {.python}
multicore = Multicore(workers = 2, multiple_iterables = True)
result = multicore.execute(function, [iterable_dataframe, iterable_list])
```

ThreshTuner
-----------

Find the best threshold to split your predictions in a binary
classification task. Most applicable for imbalance target cases. In
addition to thresholds & loss\_func scores, the predicted\_ratio
(predicted fraction of 1) will be calculated and saved for every
threshold. This will help the identify the appropriate threshold not
only based on the score, but also based on the resulting distribution of
0 and 1 in the predictions.

### Logic

> Default behavior (only pass the labels and predictions):
>
> :   -   Calculate the labels balance (fraction\_of\_1 in labels)
>     -   Define the min\_threshold as fraction\_of\_1 \* 0.8
>     -   Define the max\_threshold as fraction\_of\_1 \* 1.2 but not
>         greater than 1
>     -   Define the n\_thresholds = 200
>     -   Create 200 threshold options uniformly distributed between
>         min\_threshold & max\_threshold
>     -   Deploy the balanced\_accuracy\_score as loss\_func
>     -   Peform loss function calculation and save results in class
>         instance placeholders
>
> Customization options
>
> :   -   Change the n\_thresholds to the desired value
>     -   Change the min\_threshold & max\_threshold to the desired
>         values
>     -   Pass the loss\_func of choice, e.g. sklearn.metrics.f1\_score
>
> This will result in user defined granulation of thresholds to test

**Initialize ThreshTuner**

``` {.python}
from verstack import ThreshTuner

# initialize with default parameters
thresh = ThreshTuner()

# initialize with selected parameters
thresh = ThreshTuner(n_thresholds = 500,
                     min_threshold = 0.3,
                     max_threshold = 0.7)
```

### Parameters

-   `n_thresholds` int \[default=200\]

    Number of thresholds to test. If not set by user: 200 thresholds
    will be tested.

-   `min_threshold` float or int \[default=None\]

    Minimum threshold value. If not set by user: will be inferred from
    labels balance based on fraction\_of\_1

-   `max_threshold` float or int \[default=None\]

    Maximum threshold value. If not set by user: will be inferred from
    labels balance based on fraction\_of\_1

### Methods

-   `fit(labels, pred, loss_func)`

    Calculate loss\_func results for labels & preds for the
    defined/default thresholds. Print the threshold(s) with the best
    loss\_func scores

    > Parameters
    >
    > -   `labels` array/list/series
    >     \[default=balanced\_accuracy\_score\]
    >
    >     y\_true labels represented as 0 or 1
    >
    > -   `pred` array/list/series
    >
    >     predicted probabilities of 1
    >
    > -   `loss_func` function
    >
    >     loss function for scoring the predictions, e.g.
    >     sklearn.metrics.f1\_score

-   `result()`

    Display a dataframe with
    thresholds/loss\_func\_scores/fraction\_of\_1 for for all the the
    defined/default thresholds

-   `best_score()`

    Display a dataframe with
    thresholds/loss\_func\_scores/fraction\_of\_1 for the best
    loss\_func\_score

-   `best_predict_ratio()`

    Display a dataframe with
    thresholds/loss\_func\_scores/fraction\_of\_1 for the (predicted)
    fraction\_of\_1 which is closest to the (actual)
    labels\_fraction\_of\_1

### Examples

Use ThreshTuner with all default parameters

``` {.python}
thresh = ThreshTuner()
thres.fit(labels, pred)
```

Customized ThreshTuner application

``` {.python}
from sklearn.metrics import f1_score

thresh = ThreshTuner(n_thresholds = 500, min_threshold = 0.2, max_threshold = 0.6)
thresh.fit(labels, pred, f1_score)
```

Access the results after .fit()

``` {.python}
thresh = ThreshTuner()
thres.fit(labels, pred)

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

stratified\_continuous\_split
-----------------------------

Create stratified splits based on either continuous or categoric target variable.

:   -   For continuous target variable verstack uses binning and
        categoric split based on bins
    -   For categoric target enhanced
        sklearn.model\_selection.train\_test\_split is used: in case
        there are not enough categories for the split, the minority
        classes will be combined with nearest neighbors.

Can accept only pandas.DataFrame/pandas.Series as data input.

``` {.python}
verstack.stratified_continuous_split.scsplit(*args, 
                                             stratify, 
                                             test_size = 0.3, 
                                             train_size = 0.7, 
                                             continuous = True, 
                                             random_state = None)
```

### Parameters

-   `X,y,data`

    data input for the split in pandas.DataFrame/pandas.Series format.

-   `stratify`

    target variable for the split in pandas/eries format.

-   `test_size` \[default=0.3\]

    test split ratio.

-   `train_size` \[default=0.7\]

    train split ratio.

-   `continuous` \[default=True\]

    stratification target definition. If True, verstack will perform the
    stratification on the continuous target variable, if False,
    sklearn.model\_selection.train\_test\_split will be performed with
    verstack enhancements.

-   `random_state` \[default=5\]

    random state value.

### Examples

``` {.python}
from verstack.stratified_continuous_split import scsplit

train, test = scsplit(data, stratify = data['continuous_column_name'])
X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y, 
                                         test_size = 0.3, random_state = 5)
```

categoric\_encoders
-------------------

::: {.note}
::: {.title}
Note
:::

All the categoric encoders are conveniently integrated to work with
pandas.DataFrame. Modules receive pd.DataFrame and kwargs as inputs and
return pd.DataFrame with encoded column. All the necessary attributes
for further transform/inverse\_transform are saved in instance objects
and can be seralized (e.g. pickle) for latter application.
:::

### Factorizer

Encode categoric column by numeric labels.

#### Logic

Assign numeric labels starting with 0 to all unique variable\'s
categories.

Missing values can be encoded by an integer value (defaults to -1) /
float / string or can be left untransformed.

When transform () - unseen categories will be be represented as NaN.

**Initialize Factorizer**

``` {.python}
from verstack import Factorizer

# initialize with default parameters
factorizer = Factorizer()

# initialize with changing the NaN encoding value
factorizer = Factorizer(na_sentinel = np.nan) #-999/0.33333/'No data')
```

**Attributes**

-   `na_sentinel`

    Defined (at init) missing values encoding value.

-   `colname`

    Defined (at fit\_transform()) column that had been transformed.

-   `pattern`

    Defined (at fit\_transform()) encoding map.

#### Parameters

-   `na_sentinel` \[default=-1\]

    Missing values encoding value. Can take int/float/str/np.nan values.

#### Methods

-   `fit_transform(df, colname)`

    Fit Factorizer to data and return transformed data.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     df containing the colname to transform.
    >
    > -   `colname` str
    >
    >     Column name in df to be transformed.

-   `transform(df)`

    Apply the fitted Factorizer to new data and return transformed data.
    Unseen categories will be represented by NaN.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

-   `inverse_transform(df)`

    Inverse transform data that had been encoded by Factorizer. Data
    must contain colname that was passed at fit\_transform().

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

#### Examples

Use with default na\_sentinel:

``` {.python}
factorizer = Factorizer()
train_encoded = factorizer.fit_transform(train, 'colname') # will encode NaN values by -1
test_encoded = factorizer.transform(test)

train_reversed_to_original = factorizer.inverse_transform(train_encoded)
test_reversed_to_original = factorizer.inverse_transform(test_encoded)
```

Keep missing values untransformed:

``` {.python}
factorizer = Factorizer(na_sentinel = np.nan)
train_encoded = factorizer.fit_transform(train)
```

### OneHotEncoder

Encode categoric column by a set of binary columns.

#### Logic

Categoric \'column\':\[\'a\',\'b\',\'c\'\] will be represented by three
binary columns \'a\', \'b\', \'c\'. Original categoric \'column\' is
droped.

Missing values can be represented by a separate column or omited.

When transform() - unseen categories will not be represented by new
columns, missing categories will be represented by empty (all zeros)
columns.

**Initialize OneHotEncoder**

``` {.python}
from verstack import OneHotEncoder
ohe = OneHotEncoder()
train_encoded = ohe.fit_transform(train, 'colname') # will create a separate column for NaN values (if any)
test_encoded = ohe.transform(test)

train_reversed_to_original = ohe.inverse_transform(train_encoded)
test_reversed_to_original = ohe.inverse_transform(test_encoded)
```

**Attributes**

-   `na_sentinel`

    Defined (at init) missing values encoding value.

-   `colname`

    Defined (at fit\_transform()) column that had been transformed.

-   `categories`

    Defined (at fit\_transform()) unique class categories which will be
    represented by binary columns.

#### Parameters

-   `na_sentinel` \[default=True\]

    If True: create separate class column for NaN values.

#### Methods

-   `fit_transform(df, colname, prefix)`

    Fit OneHotEncoder to data and return transformed data.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     df containing the colname to transform.
    >
    > -   `colname` str
    >
    >     Column name in df to be transformed.
    >
    > -   `prefix` str/int/float/bool/None, optional
    >
    >     String to append DataFrame column names. The default is None.

-   `transform(df)`

    Apply the fitted OneHotEncoder to new data and return transformed
    data. Unseen categories will not be represented by new columns,
    missing categories will be represented by empty (all zeros) columns.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

-   `inverse_transform(df)`

    Inverse transform data that had been encoded by OneHotEncoder. Data
    must contain one-hot-encoded columns that was created at
    fit\_transform().

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

#### Examples

``` {.python}
ohe = OneHotEncoder()
train_encoded = ohe.fit_transform(train, 'colname', prefix = 'colname')
test_encoded = ohe.transform(test)

train_reversed_to_original = ohe.inverse_transform(train_encoded)
test_reversed_to_original = ohe.inverse_transform(test_encoded)
```

### FrequencyEncoder

Encoder to represent categoric variable classes\' frequency across the
dataset.

#### Logic

> Original column \[\'a\', \'a\', \'a\', \'b\', \'b\', \'c\', \'c\',
> \'c\', \'c\', np.nan\]
>
> Encoded column \[0.3, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.1\] \#
> np.nan\]

When transform() - unseen categories will be represented by the most
common (highest) frequency.

Can handle missing values - encode NaN by NaN frequency or leave NaN
values untransformed. Resulting frequencies are normalized as a
percentage.

**Initialize FrequencyEncoder**

``` {.python}
from verstack import FrequencyEncoder
fe = FrequencyEncoder()
train_encoded = fe.fit_transform(train, 'colname')
test_encoded = fe.transform(test)

train_reversed_to_original = fe.inverse_transform(train_encoded)
test_reversed_to_original = fe.inverse_transform(test_encoded)
```

**Attributes**

-   `na_sentinel`

    Defined (at init) missing values encoding value.

-   `colname`

    Defined (at fit\_transform()) column that had been transformed.

-   `pattern`

    Defined (at fit\_transform()) encoding map.

#### Parameters

-   `na_sentinel` \[default=True\]
    -   If True: Encode NaN values by their frequency. If False return
        np.nan in the encoded column.

#### Methods

-   `fit_transform(df, colname)`

    Fit FrequencyEncoder to data and return transformed data.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     df containing the colname to transform.
    >
    > -   `colname` str
    >
    >     Column name in df to be transformed.

-   `transform(df)`

    Apply the fitted FrequencyEncoder to new data and return transformed
    data. Unseen categories will be represented as NaN.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

-   `inverse_transform(df)`

    Inverse transform data that had been encoded by FrequencyEncoder.
    Data must contain colname that was passed at fit\_transform().

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

#### Examples

``` {.python}
frequency_encoder = FrequencyEncoder()
train_encoded = frequency_encoder.fit_transform(train, 'colname')
test_encoded = frequency_encoder.transform(test)

train_reversed_to_original = frequency_encoder.inverse_transform(train_encoded)
test_reversed_to_original = frequency_encoder.inverse_transform(test_encoded)
```

### MeanTargetEncoder

Encode train cat cols by mean target value for category.

#### Logic

To avoid target leakage train set encoding is performed by breaking data
into 5 folds & encoding categories of each fold with their respective
target mean values calculated on the other 4 folds. This will introduce
minor noize to train data encoding (at fit\_transform()) as a
normalization technique. Test set (transform()) is encoded without
normalization.

When transform() - unseen categories will be represented by the global
target mean.

Can handle missing values - encode NaN by global mean or leave NaN
values untransformed.

**Initialize MeanTargetEncoder**

``` {.python}
from verstack import MeanTargetEncoder
mean_target_encoder = MeanTargetEncoder(save_inverse_transform = True)
train_encoded = mean_target_encoder.fit_transform(train, 'colname', 'targetname')
test_encoded = mean_target_encoder.transform(test)

train_reversed_to_original = mean_target_encoder.inverse_transform(train_encoded)
test_reversed_to_original = mean_target_encoder.inverse_transform(test_encoded)
```

**Attributes**

-   `na_sentinel`

    Defined (at init) missing values encoding value.

-   `colname`

    Defined (at fit\_transform()) column that had been transformed.

-   `pattern`

    Defined (at fit\_transform()) encoding map.

-   `save_inverse_transform`

    Defined (at init) flag for saving the pattern for inverse transform.

#### Parameters

-   `na_sentinel` \[default=True\]

    If True: Encode NaN values by target global mean. If False return
    np.nan in the encoded column.

-   `save_inverse_transform` \[default=False\]

    If True: Saves mean target values for each category at each encoding
    fold. Enable if need to inverse\_transform the encoded data.
    Defaults to False because for large datasets saved pattern can
    significantly increase instance object size.

#### Methods

-   `fit_transform(df, colname, targetname)`

    Fit MeanTargetEncoder to data and return transformed data.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     df containing the colname to transform.
    >
    > -   `colname` str
    >
    >     Column name in df to be transformed.
    >
    > -   `targetname` str
    >
    >     Target column name in df for extracting the mean values for
    >     each colname category.

-   `transform(df)`

    Apply the fitted MeanTargetEncoder to new data and return
    transformed data. Unseen categories will be encoded by the global
    target mean.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

-   `inverse_transform(df)`

    Inverse transform data that had been encoded by MeanTargetEncoder.
    Data must contain colname that was passed at fit\_transform().

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

#### Examples

``` {.python}
mean_target_encoder = MeanTargetEncoder(save_inverse_transform = True)
train_encoded = mean_target_encoder.fit_transform(train, 'colname', 'targetname')
test_encoded = mean_target_encoder.transform(test)

train_reversed_to_original = mean_target_encoder.inverse_transform(train_encoded)
test_reversed_to_original = mean_target_encoder.inverse_transform(test_encoded)
```

### WeightOfEvidenceEncoder

Encoder to represent categoric variables by Weight of Evidence in
regards to the binary target variable.

#### Logic

Built on top of sclearn package
[category\_encoders.woe.WOEEncoder](https://contrib.scikit-learn.org/category_encoders/woe.html#).

If encoded value is negative - it represents a category that is more
heavily enclided to the negative target class (0). Positive encoding
result represents inclination to the positive target class (1).

When fit\_transform() is used on a train set, variable is encoded with
adding minor noize to reduce the risk of overfitting.

Can handle missing values - encode NaN by zero WoE or leave NaN
untransformed.

**Initialize WeightOfEvidenceEncoder**

``` {.python}
from verstack import WeightOfEvidenceEncoder
WOE = WeightOfEvidenceEncoder()
train_encoded = WOE.fit_transform(train, 'colname', 'targetname')
test_encoded = WOE.transform(test)

train_reversed_to_original = WOE.inverse_transform(train_encoded)
test_reversed_to_original = WOE.inverse_transform(test_encoded)
```

**Attributes**

-   `na_sentinel`

    Defined (at init) missing values encoding value.

-   `colname`

    Defined (at fit\_transform()) column that had been transformed.

-   `params`

    Defined (at init) category\_encoders.woe.WOEEncoder
    [parameters](https://contrib.scikit-learn.org/category_encoders/woe.html#)

#### Parameters

-   `na_sentinel` \[default=True\]

    If True: Encode NaN values by zero WoE. If False return np.nan in
    the encoded column.

-   `kwargs`

    category\_encoders.woe.WOEEncoder
    [parameters](https://contrib.scikit-learn.org/category_encoders/woe.html#).
    Following parameters are set by default: `'randomized':True`,
    `'random_state':42`, `'handle_missing':'return_nan'` \<- inferred
    from na\_sentinel setting.

#### Methods

-   `fit_transform(df, colname, targetname)`

    Fit WeightOfEvidenceEncoder to data and return transformed data.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     df containing the colname to transform.
    >
    > -   `colname` str
    >
    >     Column name in df to be transformed.
    >
    > -   `targetname` str
    >
    >     Target column name in df for calculating WoE for each colname
    >     category.

-   `transform(df)`

    Apply the fitted WeightOfEvidenceEncoder to new data and return
    transformed data. Unseen categories\' WoE is set to 0.

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

-   `inverse_transform(df)`

    Inverse transform data that had been encoded by
    WeightOfEvidenceEncoder. Data must contain colname that was passed
    at fit\_transform().

    > Parameters
    >
    > -   `df` pd.DataFrame
    >
    >     Data containing the colname to transform.

#### Examples

``` {.python}
WOE = WeightOfEvidenceEncoder()
train_encoded = WOE.fit_transform(train, 'colname', 'targetname')
test_encoded = WOE.transform(test)

train_reversed_to_original = WOE.inverse_transform(train_encoded)
test_reversed_to_original = WOE.inverse_transform(test_encoded)
```

timer
-----

Timer decorator to measure any function execution time and create
elapsed time output: hours/minues/seconds will be calculated and
returned conveniently.

``` {.python}
verstack.tools.timer
```

### Examples

timer is a decorator function: it must placed above the function (that
needs to be timed) definition

``` {.python}
from verstack.tools import timer

@timer
def func(a,b):
    print(f'Result is: {a + b}')

func(2,3)

>>>Result is: 5
>>>Time elapsed for func execution: 0.0002 seconds
```

pretty_print
------------

Function to execute print statements subject to verbose argument and order of printed message. 

``` {.python}
verstack.tools.pretty_print
```

## Examples

Add print statements to your program with different level of indentation for different messages and have them printed subject on the global verbosity setting in your program. A convenient way to set up verbosity for large projects without having to define all the print statements with ``if verbose == True``. Just pass the verbose argument to the pretty_print function.

``` {.python}
from verstack.tools import pretty_print

# define a function/program/code

def do_something(a, b, c, verbose):
pretty_print('Executing do_something() function', order = 0, verbose = verbose)

result_1 = a + b
pretty_print(f'A + B result is {result_1}', order = 1, verbose = verbose)

try:
    a / b:
except ZeroDivisionError:
    pretty_print('Argument b can not be zero', order = 2, verbose = verbose)

result_2 = b + calculated
pretty_print(f'B + C result is {result_2}', order = 1, verbose = verbose)

pretty_print('do_something() function execution completed', order = 1, verbose = verbose)

do_something(1,0,5, verbose = False)
# no output to the console

do_something(1,0,5, verbose = True)

>>> ----------------------------------------------------------------------
>>> Executing do_something() function
>>> ----------------------------------------------------------------------

>>> - A + B result is 1
>>>   . Argument b can not be zero

>>> - B + C result is 5

>>> - do_something() function execution completed
```

Links
-----

[Git](https://github.com/DanilZherebtsov/verstack)

[pypi](https://pypi.org/project/verstack/)

[author](https://www.linkedin.com/in/danil-zherebtsov/)
