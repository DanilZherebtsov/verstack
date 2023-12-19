import pandas as pd
import sys

def is_bool_na_sentinel(val):
    if isinstance(val, bool):
        return val
    else:
        print('na_sentiment must be bool. Continuing with default na_sentinel = True')
        return True
# -----------------------------------------------------------------------------

def is_not_bool_na_sentinel(val):
    if isinstance(val, bool):
        print('na_sentiment must int/float/string/np.nan but not bool. Continuing with default na_sentinel = -1')
        return -1
    else:
        return val    
# -----------------------------------------------------------------------------

def assert_fit_transform_args(df, colname, targetname = None):
    # df
    try:
        assert(isinstance(df, pd.DataFrame))
    except:
        raise TypeError('First argument to fit_transform() must be a pd.DataFrame')
    # colname
    try:
        assert(isinstance(colname, str))
    except: 
        raise TypeError('Second argument to fit_transform() must be a string')
    try:
        assert(colname in df)
    except:
        raise KeyError('Second argument to fit_transform() must a valid column name in df')
        
    # targetname
    if targetname:
        try:
            assert(isinstance(targetname, str))
        except:
            raise TypeError('Third argument to fit_transform() must be a string')
        try:
            assert(targetname in df)
        except:
            raise KeyError('"targetname" must a valid column name in df')
# -----------------------------------------------------------------------------
        
def assert_transform_args(df):
    try:
        assert(isinstance(df, pd.DataFrame))
    except:
        raise TypeError('Only argument to transform()/inverse_transform() must be a pd.DataFrame')
# -----------------------------------------------------------------------------
        
def assert_binary_target(df, targetname):
    try:
        assert(df[targetname].nunique() == 2)
    except:
        raise ValueError(f'The target variable must be binary, instead in contains {df[targetname].nunique()} unique values')
# -----------------------------------------------------------------------------
