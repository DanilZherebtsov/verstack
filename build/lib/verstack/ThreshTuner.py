import operator
import numpy as np
import pandas as pd
from collections import Counter

class ThreshTuner():
    
    __version__ = '0.0.4'
    
    ''' Tune threshold for binary classificaiton models output. '''
    
    def __init__(self, n_thresholds = 200, min_threshold = None, max_threshold = None, verbose = True):
        '''
        Initialize class instance

        Parameters
        ----------
        n_thresholds : int, optional
            n_thresholds will be uniformly distributed between
            max_threshold and min_threshold. The default is 200.
        min_threshold : float/int, optional
            Minimum border of thresholds range. If not set, will be inferred automatically.
            The default is None.
        max_threshold : float/int, optional
            Minimum border of thresholds range. If not set, will be inferred automatically.
            The default is None.

        Returns
        -------
        None.

        '''
        self.n_thresholds = n_thresholds
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.labels_fraction_of_1 = None
        self.loss_func = None
        self.result = None
        self.verbose = verbose
        if self.verbose:
            print(self.__repr__())
        
    # print init parameters when calling the class instance
    def __repr__(self):
        return f'ThreshTuner(n_threshols = {self.n_thresholds},\
            \n            min_threshold = {self.min_threshold},\
                \n            max_threshold = {self.max_threshold},\
                    \n            labels_fraction_of_1 = {self.labels_fraction_of_1},\
                        \n            loss_func = {self.loss_func.__name__ if self.loss_func else None}'

    # Validate init arguments
    # =========================================================
    # n_thresholds
    n_thresholds = property(operator.attrgetter('_n_thresholds'))

    @n_thresholds.setter
    def n_thresholds(self, n):
        if type(n) != int : raise Exception('n_thresholds must be integer')
        if type(n) == int and n<1 : raise Exception('n_thresholds must be greater than 0')
        self._n_thresholds = n

    # -------------------------------------------------------
    # min_threshold
    min_threshold = property(operator.attrgetter('_min_threshold'))
    
    @min_threshold.setter
    def min_threshold(self, min_t):
        if min_t:
            if type(min_t) not in [np.float64, float] and type(min_t) != int : raise Exception('min_threshold must be integer or float')
            if type(min_t) in [np.float64, float] or type(min_t) == int:
                if min_t < 0 or min_t >= 1 : raise Exception('min_threshold must be greater than 0 and less than 1')

        self._min_threshold = min_t

    # -------------------------------------------------------
    # max_threshold
    max_threshold = property(operator.attrgetter('_max_threshold'))
    
    @max_threshold.setter
    def max_threshold(self, max_t):
        if max_t:
            if type(max_t) not in [np.float64, float] and type(max_t) != int : raise Exception('max_threshold must be integer or float')
            if type(max_t) in [np.float64, float] or type(max_t) == int:
                if max_t <= 0 or max_t > 1 : raise Exception('max_threshold must be greater than 0 and less than or equal to 1')
            if self.min_threshold:
                if max_t < self.min_threshold : raise Exception('max_threshold must be grater than min_threshold')
        self._max_threshold = max_t

    # =========================================================

    def _get_thresh_barriers(self, labels, smallest = True):
        '''
        Extend threshold barriers by 20% for a more comprehensive search.
        
        Parameters
        ----------
        labels : pd.Series
            labels.
        smallest : bool, optional
            Indicator for min_threshold/max_threshold calculation. 
            The default is True.

        Returns
        -------
        thresh : float
            extended threshold barrier value.

        '''
        if smallest:
            thresh = labels.value_counts(normalize = True).min()*0.8
        else:
            thresh = labels.value_counts(normalize = True).max()*1.2
            if thresh > 1:
                thresh = int(thresh)
        return thresh

    def _measure_metrics(self, labels, pred, thresholds):
        '''
        Calculate score for each threshold.
        
        Create pd.DataFrame with thresholds, scores and fraction_of_1.
        Save result in the instance variable self.result.

        Parameters
        ----------
        labels : pd.Series
            labels.
        pred : pd.Series
            predicted probabilities of 1.
        thresholds : array
            thresholds to use for calculating score.

        Returns
        -------
        None.

        '''
        resulting_dict = {}
        for i in thresholds:
            if True in dict(Counter(pred>i)).keys():
                result_temp = self.loss_func(labels, pred>i)
                resulting_dict[i] = {}
                resulting_dict[i][self.loss_func.__name__] = result_temp
                resulting_dict[i]['fraction_of_1'] = dict(Counter(pred>i))[True]/len(pred)
        result = pd.DataFrame(resulting_dict).T
        # rearange dataframe placing index into 'threshold' column
        result['threshold'] = result.index
        result.reset_index(drop = True, inplace = True)
        cols = [result.columns[-1]] + result.columns[:-1].tolist()
        result = result[cols]        
        self.result = result            

    def best_score(self):
        ''' Print results with the best metric value.'''
        return self.result[self.result[self.loss_func.__name__] == self.result[self.loss_func.__name__].max()]

    def best_predict_ratio(self):
        ''' Print results with the best prediction fraction of 1.
        
        Best == closest to labels fraction of 1.
        
        '''        
        diffs = abs(self.result['fraction_of_1'] - self.labels_fraction_of_1)
        ix = diffs[diffs == diffs.min()].index[0]
        return pd.DataFrame(dict(zip(self.result.loc[ix].index.tolist(), self.result.loc[ix].values)), index = [ix])

    def fit(self, labels, pred, loss_func = None):
        '''
        Measure score according to loss_func for labels and predictions.
        
        Create a uniform distribution of thresholds between min_threshold 
        and max_threshold values.
        Measure scores.
        Print best threshold(s) results.

        Parameters
        ----------
        labels : array/list/pd.Series
            labels. Must be represented by 0 and 1 with 1 being the positive class.
        pred : array/list/pd.Series
            predicted probabilities of 1.
        loss_func : function, optional
            loss function to measure the score. If None is passed, balanced_accuracy_score
            is used for threshold tuning.

        Raises
        ------
        Exception
            if labels are not represented by 0 or 1.
            if labels shape is inconsistent with pred shape.
            

        Returns
        -------
        None.

        '''

        from sklearn.metrics import balanced_accuracy_score

        labels = pd.Series(labels)
        pred = pd.Series(pred)

        # Validate arguments
        # =========================================================        
        if np.any([x not in [0,1] for x in labels.unique()]):
            raise Exception('labels must be represented by 0 and 1')
        # ---------------------------------------------------------
        if labels.shape != pred.shape:
            raise Exception(f'Labels shape {labels.shape} is inconsistent with predictions shape {pred.shape}')
        # =========================================================        

        if not self.min_threshold:
            self.min_threshold = self._get_thresh_barriers(labels, smallest = True)
        if not self.max_threshold:
            self.max_threshold = self._get_thresh_barriers(labels, smallest = False)

        if loss_func:
            self.loss_func = loss_func
        else:
            self.loss_func = balanced_accuracy_score

        self.labels_fraction_of_1 = labels.value_counts()[1]/len(labels)

        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)

        self._measure_metrics(labels, pred, thresholds)

        if self.verbose:
            print(f'\n                   Best threshold(s)\n{"-"*55}')
            # print result without index
            print(self.result[self.result[self.loss_func.__name__] == self.result[self.loss_func.__name__].max()].to_string(index = False))
            print('-'*55)


