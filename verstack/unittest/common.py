import numpy as np
import pandas as pd
import random

binary_target_choices = [0,1]
multiclass_target_choices = [0,1,2,3]
cat_col_choices = ['a', 'b', 'c', 'd']
date_col_choices = [
    '2015-06-17',
    '2015-05-31',
    '2015-08-28',
    '2015-06-29',
    '2015-03-17',
    '2015-02-25',
    '2015-01-12',
    '2015-11-29',
    '2015-12-29',
    '2015-12-12',
    '2015-12-29',
    '2015-09-30',
    '2015-07-01',
    '2015-06-30',
    '2015-05-30',
    '2015-04-23',
    '2015-07-25',
    '2015-04-30',
    '2015-07-31'
    ]

def generate_data(processed=False):
    # generate data either unprocessed or processed (ready for modeling)
    np.random.seed(42)
    random.seed(42)
    if not processed:
        var_x = [random.choice(cat_col_choices) for i in range(100)]
        var_timestamp = [random.choice(date_col_choices) for i in range(100)]
        numeric_target = list(np.random.uniform(0,1,100))
        binary_target = [random.choice(binary_target_choices) for i in range(100)]
        df = pd.DataFrame({'x':var_x, 'y':numeric_target, 'y_binary':binary_target, 'timestamp':var_timestamp})
        df.loc[2,'x'] = np.nan
        df.loc[3,'timestamp'] = np.nan
    else:
        var_x = list(np.random.uniform(0,1,100))
        var_z = list(np.random.uniform(0,1,100))
        numeric_target = list(np.random.uniform(0,1,100))
        binary_target = [random.choice(binary_target_choices) for i in range(100)]
        multiclass_target = [random.choice(multiclass_target_choices) for i in range(100)]
        df = pd.DataFrame({'x':var_x, 'z':var_z, 'y':numeric_target, 'y_binary':binary_target, 'y_multiclass':multiclass_target})
    return df

