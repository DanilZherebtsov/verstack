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

complex_date_col_choices = [
    '2011-11-11 21:44:35 UTC',
    '2012-04-23 23:22:00 UTC',
    '2011-04-05 08:30:00 UTC',
    '2009-10-12 21:40:00 UTC',
    '2009-07-26 03:07:00 UTC',
    '2014-03-21 08:20:00 UTC',
    '2014-07-19 02:46:00 UTC',
    '2013-07-05 12:03:00 UTC',
    '2010-07-10 16:09:00 UTC',
    '2012-07-12 05:48:36 UTC',
    '2014-10-18 12:40:00 UTC',
    '2014-11-16 15:36:00 UTC',
    '2009-02-25 09:04:00 UTC',
    '2014-10-10 15:10:00 UTC',
    '2011-01-19 20:54:00 UTC',
    '2013-12-03 21:50:05 UTC',
    '2014-12-27 03:28:20 UTC',
    '2010-01-19 15:54:56 UTC',
    '2009-03-26 11:49:46 UTC',
    '2010-12-19 13:11:06 UTC',
    '2009-02-24 08:29:00 UTC'
    ]

def generate_data(processed=False):
    # generate data either unprocessed or processed (ready for modeling)
    np.random.seed(42)
    random.seed(42)
    if not processed:
        var_x = [random.choice(cat_col_choices) for i in range(100)]
        var_date = [random.choice(date_col_choices) for i in range(100)]
        var_timestamp = [random.choice(complex_date_col_choices) for i in range(100)]
        numeric_target = list(np.random.uniform(0,1,100))
        binary_target = [random.choice(binary_target_choices) for i in range(100)]
        df = pd.DataFrame({'x':var_x, 'y':numeric_target, 'y_binary':binary_target, 'date':var_date, 'timestamp':var_timestamp})
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

