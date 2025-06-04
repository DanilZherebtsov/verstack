import pytest
import pandas as pd
import numpy as np
import sys

sys.path.append("../")
from DateParser import DateParser

def test_all_date_formats_and_transform():
    """
    Test that DateParser correctly handles all supported date formats
    and properly implements the transform method.
    """
    date_formats = {
        'iso_format': ['2023-02-01', '2023-01-02', '2022-01-03', '2023-01-04', '2022-01-05'],
        'slash_format': ['2023/01/01', '2023/01/02', '2023/01/03', '2023/01/04', '2023/01/05'],
        'dot_format': ['2023.01.01', '2023.01.02', '2023.01.03', '2023.01.04', '2023.01.05'],
        
        'dmy_format': ['01-01-2023', '02-01-2023', '03-01-2023', '04-01-2023', '05-01-2023'],
        'mdy_format': ['01-31-2023', '02-28-2023', '03-31-2023', '04-30-2023', '05-31-2023'],
        'ymd_format': ['2023-01-23', '2023-02-24', '2023-03-11', '2023-04-01', '2023-05-02'],
        
        'datetime_hm': ['2023-01-01 13:45', '2023-01-02 14:30', '2023-01-03 15:15', '2023-01-04 16:00', '2023-01-05 17:45'],
        'datetime_hms': ['2023-01-01 13:45:30', '2023-01-02 14:30:45', '2023-01-03 15:15:00', '2023-01-04 16:00:15', '2023-01-05 17:45:30'],
        
        'datetime_tz_named': ['2023-01-01 13:45 UTC', '2023-01-02 14:30 GMT', '2023-01-03 15:15 EDT', '2023-01-04 16:00 CET', '2023-01-05 17:45 PST'],
        'datetime_tz_offset': ['2023-01-01 13:45 +0000', '2023-01-02 14:30 -0500', '2023-01-03 15:15 +0300', '2023-01-04 16:00 +0100', '2023-01-05 17:45 -0800'],
        'datetime_tz_colon': ['2023-01-01 13:45 +00:00', '2023-01-02 14:30 -05:00', '2023-01-03 15:15 +03:00', '2023-01-04 16:00 +01:00', '2023-01-05 17:45 -08:00'],
        
        'with_errors': ['2023-01-01', '2023-01-02', 'not_a_date', '2023-01-04', 'also_not_a_date'],
        
        'not_date': ['apple', 'banana', 'cherry', 'strawberry', 'kiwi']
    }
    
    train_data = {col: values[:3] for col, values in date_formats.items()}
    train_df = pd.DataFrame(train_data)
    
    # Create test DataFrame with different but compatible values
    test_data = {col: values[3:] for col, values in date_formats.items()}
    test_df = pd.DataFrame(test_data)
    
    # Initialize DateParser with error tolerance to handle the 'with_errors' column
    parser = DateParser(verbose=False)
    
    transformed_train = parser.fit_transform(train_df, error_tolerance=0.1)
    transformed_test = parser.transform(test_df)

    # Verify datetime columns were detected correctly with error_tolerance = 0.1 shold not parse 'with_errors'
    expected_datetime_cols = [
        'iso_format', 'slash_format', 'dot_format', 'dmy_format', 'mdy_format',
        'datetime_hm', 'datetime_hms', 'datetime_tz_named', 'datetime_tz_offset', 
        'datetime_tz_colon', 'ymd_format'
    ]
    for col in expected_datetime_cols:
        assert col in parser.datetime_cols, f"Failed to detect {col} as datetime"
    # ------------------------------------------------------------------------------------
    # check error_tolerance
    expected_datetime_cols.append('with_errors')
    parser = DateParser(verbose=False)
    transformed_train = parser.fit_transform(train_df, error_tolerance=0.5)
    transformed_test = parser.transform(test_df)
    assert 'with_errors' in parser.datetime_cols
    # ------------------------------------------------------------------------------------
    # Check 'not_date' was not identified as datetime
    assert 'not_date' not in parser.datetime_cols
    # ------------------------------------------------------------------------------------
    # Verify at least some basic features were extracted
    assert 'year' in parser.created_datetime_cols['iso_format']
    assert 'month' in parser.created_datetime_cols['iso_format']
    assert 'day' in parser.created_datetime_cols['iso_format']

    assert 'hour' in parser.created_datetime_cols['datetime_hm']
    assert 'minute' in parser.created_datetime_cols['datetime_hm']
    assert 'part_of_day' in parser.created_datetime_cols['datetime_hm']
    assert 'second' in parser.created_datetime_cols['datetime_hms']
    assert 'minute' in parser.created_datetime_cols['datetime_tz_offset']
    
    # Check transform consistency - the columns in transformed_train and transformed_test should match
    assert set(transformed_train.columns) == set(transformed_test.columns)
    
    # Verify original datetime columns were removed
    for col in expected_datetime_cols:
        assert col not in transformed_train.columns
        assert col not in transformed_test.columns
    
    # Verify 'not_date' still exists
    assert 'not_date' in transformed_train.columns
    assert 'not_date' in transformed_test.columns
    
    # Check feature values for a specific column
    assert 'iso_format_year' in transformed_test.columns
    assert transformed_test['iso_format_year'].iloc[0] == 2023

    # check ymd_format
    assert transformed_train['ymd_format_month'].iloc[0] == 1
    assert transformed_train['ymd_format_day'].iloc[0] == 23
    

def test_transform_specific_features():
    """
    Test the correct generation of specific datetime features
    during transform operation
    """
    # Create sample data with clear differences in day, month, hour, etc.
    train_df = pd.DataFrame({
        'datetime': [
            '2023-01-01 09:30:00',
            '2023-02-15 14:45:00'
        ]
    })
    
    test_df = pd.DataFrame({
        'datetime': [
            '2023-03-10 18:15:00',
            '2023-04-20 22:30:00'
        ]
    })
    
    parser = DateParser(verbose=False)
    transformed_train = parser.fit_transform(train_df)
    transformed_test = parser.transform(test_df)
    
    # Verify specific features
    expected_features = [
        'datetime_month', 'datetime_week', 'datetime_day', 
        'datetime_hour', 'datetime_minute', 'datetime_weekday',
        'datetime_part_of_day'
    ]
    for feature in expected_features:
        assert feature in transformed_test.columns
    
    # Check specific values in test data
    assert transformed_test['datetime_month'].iloc[0] == 3  # March
    assert transformed_test['datetime_day'].iloc[0] == 10
    assert transformed_test['datetime_hour'].iloc[0] == 18
    
    # Check part_of_day feature calculation
    # 18:15 should be part_of_day 3 (evening: 18-22)
    assert transformed_test['datetime_part_of_day'].iloc[0] == 3
    
    # 22:30 should be part_of_day 3 (evening: 18-22)
    assert transformed_test['datetime_part_of_day'].iloc[1] == 3