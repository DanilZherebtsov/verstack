"""
Created on Wed Feb 09 15:01:43 2022

@author: Danil Zherebtsov
"""

'''
further improvements:
    - use locale of the browser and create is_holyday feature
        - automate holidays feats creation based on locale
'''

import numpy as np
import pandas as pd
import sys
import gc
import holidays
from datetime import date, datetime
import dateutil.parser as parser
from verstack.tools import Printer

DATE_DELIMITERS = ['.', '-', '/']
# -----------------------------------------------------------------------------
formats = [
    '28-OCT-90',
    '28-OCT-1990',
    '10/28/90',
    '10/28/1990',
    '28.10.90',
    '28.10.1990',
    '90/10/28',
    '1990/10/28',
    '4 Q 90',
    '4 Q 1990',
    'OCT 90',
    'OCT 1990',
    '43 WK 90',
    '43 WK 1990',
    '01:02',
    '02:34',
    '02:34.75',
    '20-JUN-1990 08:03',
    '20-JUN-1990 08:03:00',
    '1990-06-20 08:03',
    '1990-06-20 08:03:00.0'
    ]

# -----------------------------------------------------------------------------
months = {'JAN':1,
          'FEB':2,
          'MAR':3,
          'APR':4,
          'MAY':5,
          'JUN':6,
          'JUL':7,
          'AUG':8,
          'SEP':9,
          'OCT':10,
          'NOV':11,
          'DEC':12}
# -----------------------------------------------------------------------------
weeks = [f'{str(wk)} WK' for wk in range(1,53)]
yeardays = [wk*7 for wk in range(1,53)]

week_yearday = dict(zip(weeks, yeardays))

month_days = {1:31,
              2:30,
              3:31,
              4:30,
              5:31,
              6:30,
              7:31,
              8:31,
              9:30,
              10:31,
              11:30,
              12:31}
# -----------------------------------------------------------------------------
quarters = {'1 Q': '3/31',
            '2 Q': '6/30',
            '3 Q': '9/30',
            '4 Q': '12/31'}
# -----------------------------------------------------------------------------
states_provinces_dict = {
    'prov': [
        'Australia', 'AU', 'AUS',
        'Austria', 'AT', 'AUT',
        'Canada', 'CA', 'CAN',
        'France', 'FR', 'FRA',
        'Germany', 'DE', 'DEU',
        'India', 'IN', 'IND',
        'Italy', 'IT', 'ITA',
        'NewZealand', 'NZ', 'NZL',
        'Nicaragua', 'NI', 'NIC',
        'Spain', 'ES', 'ESP',
        'Switzerland', 'CH', 'CHE' 
        ],
    'state': [
        'Brazil', 'BR', 'BRA',
        'Chile', 'CL', 'CHL'
        'Malaysia', 'MY', 'MYS',
        'UnitedKingdom', 'UK', 'GB', 'GBR',
        'UnitedStates', 'US', 'USA'
        ]
    }
# -----------------------------------------------------------------------------
class DateParser():
    
    __version__ = '0.0.8'

    def __init__(self, country = None, state = None, prov = None, payday = None, verbose = True):
        '''
        Initialize DateParser instance.

        Create empty blank attributes to be filled if datetime cols are found:
            datetime_cols = None
            created_datetime_cols = None
        
        Parameters
        ----------
        country : str, optional
            Country name for parsing holidays. The default is None.
        state : str, optional
            State name (specific to country) for parsing holidays. The default is None.
        prov : str, optional
            State name (specific to country) for parsing holidays. The default is None.
        Note: valid country/state/province arguments are available at: https://pypi.org/project/holidays/
        
        Returns
        -------
        None.

        '''
        self.verbose = verbose
        self.printer = Printer(verbose=self.verbose)
        self.country = country
        self.state = state
        self.prov = prov
        self.payday = payday
        self._datetime_cols = []
        self._created_datetime_cols = []
        self.supported_formats = formats

    
    # -----------------------------------------------------------------------------
    # print init parameters when calling the class instance
    def __repr__(self):
        return f'DateParser(country: {self.country}\
            \n           state: {self.state}\
            \n           prov: {self.prov}\
            \n           payday: {self.payday}\
            \n           datetime_cols: {self._datetime_cols}\
            \n           created_datetime_cols: {self._created_datetime_cols}\
            \n           verbose: {self.verbose}'

    # Validate init arguments
    # =========================================================================
    def _print_warning(self):
        supported_countries = holidays.list_supported_countries()
        for country in supported_countries:
            self.printer.print(country, order=5, force_print=True)

    # country
    @property
    def country(self):
        return self._country

    @country.setter
    def country(self, value):
        supported_countries = holidays.list_supported_countries()
        if value:
            if not isinstance(value, str):
                self._print_warning()
                raise TypeError('country name must be a string')
            if value not in supported_countries:
                self._print_warning()
                raise ValueError(f'{value} is not a valid country name\nPlease enter a valid country name from the above list\nSupported countries list can be obtained by date_parser_instance.list_supported_countries() function')
            self._country = value
        else:
            self._country = None
    # -------------------------------------------------------------------------
    # state
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state:
            if not isinstance(state, str):
                raise TypeError('state argument must be a string')
            if not self.country:
                raise ValueError('country argument must be passed')
            if self.country in states_provinces_dict['state']:
                self._state = state
            else:
                self._state = None
        else:
            self._state = None
    # -------------------------------------------------------------------------
    # prov
    @property
    def prov(self):
        return self._prov

    @prov.setter
    def prov(self, prov):
        if prov:
            if not isinstance(prov, str):
                raise TypeError('prov (province) argument must be a string')
            if not self.country:
                raise ValueError('country argument must be passed')
            if self.country in states_provinces_dict['prov']:
                self._prov = prov
            else:
                self._prov = None
        else:
            self._prov = None
    # -----------------------------------------------------------------------------
    # payday
    @property
    def payday(self):
        return self._payday

    @payday.setter
    def payday(self, payday):
        if payday:
            if not isinstance(payday, list):
                raise TypeError('payday argument must be a list of integers')
            if not np.all([type(x) == int for x in payday]):
                raise ValueError('payday argument must be a list of integers. E.g. [1,15]')
            self._payday = payday
        else:
            self._payday = None
    # -----------------------------------------------------------------------------
    # datetime_cols
    @property
    def datetime_cols(self):
        return self._datetime_cols
    # -----------------------------------------------------------------------------
    # created_datetime_cols
    @property
    def created_datetime_cols(self):
        return self._created_datetime_cols
    # -----------------------------------------------------------------------------
    # verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if not isinstance(verbose, bool):
            self.printer.print('verbose argument must be bool (True/False). Setting default True', order='error')
            self._verbose = True
        else:
            self._verbose = verbose
    # =========================================================================
    # Validate methods arguments
    def _validate_country(self, value):
        supported_countries = holidays.list_supported_countries()
        if not isinstance(value, str):
            self._print_warning()
            raise TypeError('country name must be a string')
        if value not in supported_countries:
            self._print_warning()
            raise ValueError(f'{value} is not a valid country name\nPlease enter a valid country name from the above list\nSupported countries list can be obtained by date_parser_instance.list_supported_countries() function')
        return value
    # -------------------------------------------------------------------------
    def _validate_state(self, country, state):
        if state:
            if not isinstance(state, str):
                raise TypeError('state argument must be a string')
            if not self.country:
                raise ValueError('country argument must be passed')
            if self.country in states_provinces_dict['state']:
                return state
        else:
            return None
    # -------------------------------------------------------------------------
    def _validate_prov(self, country, prov):
        if prov:
            if not isinstance(prov, str):
                raise TypeError('prov (province) argument must be a string')
            if not self.country:
                raise ValueError('country argument must be passed')
            if self.country in states_provinces_dict['prov']:
                return prov
        else:
            return None
    # =========================================================================
    def _extract_years(self, datetime_col_series):
        '''
        Extract all unique years from datetime_col_series

        Parameters
        ----------
        datetime_col_series : pd.Series of type pd.Datetime
            series with datetime objects

        Returns
        -------
        years : list
            list of unique years.

        '''
        years = list(datetime_col_series.dropna().dt.year.unique())
        years = [int(year) for year in years]
        return years
    # -----------------------------------------------------------------------------
    def _return_holidays_flag(self, date, holidays_calendar):
        '''
        Validate if date string is in holidays_calendar

        Parameters
        ----------
        date : str
            datetime object.
        holidays_calendar : holidays.instance
            holidays dictionary.

        Returns
        -------
        bool
            True if date is holiday, False if not.

        '''
        return date in holidays_calendar
    # -----------------------------------------------------------------------------
    def _return_holidays_names(self, date, holidays_calendar):
        '''
        Validate if date string is in holidays_calendar and return holiday name

        Parameters
        ----------
        date : str
            datetime object.
        holidays_calendar : holidays.instance
            holidays dictionary.

        Returns
        -------
        str
            valid holiday name or 'not_a_holiday'

        '''
        if date in holidays_calendar:
            return holidays_calendar[date]
        else:
            return 'not_a_holiday'
    # -----------------------------------------------------------------------------
    def list_supported_countries(self):
        '''
        Print holidays package supported countries

        Returns
        -------
        None.

        '''
        print(holidays.list_supported_countries())
    # -----------------------------------------------------------------------------
    def get_holidays_calendar(self, country, years, state = None, prov = None):
        '''
        Create holidays calendar for a given country, years, state or province

        Parameters
        ----------
        country : str
            country name.
        years : list
            list of years to parse holidays.
        state : str, optional
            state name. The default is None.
        prov : str, optional
            province name. The default is None.
            
        Note: valid country/state/province arguments are available at: https://pypi.org/project/holidays/

        Returns
        -------
        holidays_calendar : holidays instance
            list of holidays dates and names.

        '''
        country = self._validate_country(country)
        state = self._validate_state(country, state)
        prov = self._validate_prov(country, prov) 
        holidays_calendar = holidays.CountryHoliday(country, years, state = state, prov = prov)
        return holidays_calendar        
    # -----------------------------------------------------------------------------
    def parse_holidays(self, datetime_col_series, country, state = None, prov = None, holiday_names = False):
        '''
        Extract holidays from a series of datetimes

        Parameters
        ----------
        datetime_col_series : pd.Series with dtypes in ['O', 'pd.Datetime']
            datetime objects.
        country : str
            country name.
        state : str, optional
            state name. The default is None.
        prov : str, optional
            province name. The default is None.
        holiday_names : bool, optional
            If True: returns individual holidays name, else: returns 
            binary flag (0 or 1) if date is a holiday. 
            The default is False.

        Raises
        ------
        TypeError
            If passed .

        Returns
        -------
        pd.Series
            holidays data.

        '''
        if datetime_col_series.dtype == 'O':
            try:
                datetime_col_series = pd.to_datetime(datetime_col_series)
            except:
                raise TypeError('Data can not be converted to datetime format')
        years = self._extract_years(datetime_col_series)            
        holidays_calendar = self.get_holidays_calendar(country, years, state, prov)
        if holiday_names:    
            holidays_names_series = datetime_col_series.dropna().apply(self._return_holidays_names, holidays_calendar = holidays_calendar)
            return holidays_names_series
        else:
            holidays_flags_series = datetime_col_series.dropna().apply(self._return_holidays_flag, holidays_calendar = holidays_calendar)
            holidays_flags_series = holidays_flags_series.astype(int)
            return holidays_flags_series
    # -----------------------------------------------------------------------------
    def _get_days_from_epoch(self, datetime_object):
        '''
        Get number of days between datetime_object and epoch (1970-01-01).

        Parameters
        ----------
        datetime_object : pd.Datetime
            datetime value.

        Returns
        -------
        int
            number of days since epoch.

        '''
        return (datetime_object.tz_localize(None) - datetime(1970,1,1)).days
    # -----------------------------------------------------------------------------
    def _get_payday_flag(self, day):
        '''
        Evaluate date (day) as a payday or not.

        Parameters
        ----------
        day : int
            monthday.

        Returns
        -------
        int
            0 if not a payday else 1.

        '''
        payday = self.payday
        if day in payday:
            return 1
        else:
            return 0
    # -----------------------------------------------------------------------------
    def _get_unit_contents(self, datetime_series, unit):
        '''Extract year / month / day array from datetime_series'''
        if unit == 'year':
            unit_contents = datetime_series.dt.year
        elif unit == 'quarter':
            unit_contents = datetime_series.dt.quarter
        elif unit == 'month':
            unit_contents = datetime_series.dt.month
        elif unit == 'week':
            unit_contents = datetime_series.dt.week
        elif unit == 'day':
            unit_contents = datetime_series.dt.day
        elif unit == 'dayofyear':
            unit_contents = datetime_series.dt.dayofyear
        elif unit == 'weekday':
            unit_contents = datetime_series.dt.weekday
        elif unit == 'hour':
            unit_contents = datetime_series.dt.hour
        elif unit == 'minute':
            unit_contents = datetime_series.dt.minute
        elif unit == 'second':
            unit_contents = datetime_series.dt.second
        return unit_contents
    # -----------------------------------------------------------------------------
    def _extract_default_feats(self, X, col, train, prefix=''):
        """
        Create new features based on datetime column.

        Extract information from datetime column and create corresponding date unit
        features.

        Args:
            X (pd.DataFrame): data with datetime column(s).
            col (str): datetime column name.
            train (bool): Apply function to train or test set.
                If True, instance attribute self._created_datetime_cols
                gets appended with new column 'part_of_day'
            prefix (str, optional): prefix to the new features names. Defaults to ''.

        Returns: None

        """

        calendar_units = ['year', 'week', 'day']

        time_units = ['quarter', 'month', 'weekday', 'dayofyear', 'hour', 
                      'minute', 'second']

        temp = X.sample(100 if len(X) >= 100 else len(X))

        for unit in calendar_units:
            unit_contents = self._get_unit_contents(X[col], unit)
            if np.any(unit_contents):
                X[prefix+unit] = unit_contents
                try:
                    X[prefix+unit] = X[prefix+unit].astype(int)
                except ValueError: # handle NaN
                    X[prefix+unit] = X[prefix+unit].astype(float)
                if train:
                    self._created_datetime_cols.append(prefix+unit)

        for unit in time_units:
            unit_contents = self._get_unit_contents(X[col], unit)
            if np.any(unit_contents):
                X[prefix+unit] = unit_contents
                if train:
                    self._created_datetime_cols.append(prefix+unit)
        return 0

    def extract_time_of_day(self, hour):
        """
        Find part of day based on hour.

        Args:
            hour (int): hour value.
            train (bool): Apply function to train or test set.
                If True, instance attribute self._created_datetime_cols
                gets appended with new column 'part_of_day'
        Returns:
            (int): value of calculated part of day.

        """
        if 6 <= hour <= 11:
            return 1
        elif 12 <= hour <= 17:
            return 2
        elif 18 <= hour <= 22:
            return 3
        else:
            return 4

    def _extract_all_feats(self, X, train=True):
        """
        Extract all the supporting datetime features from all datetime columns.

        Apply functions:
            - _extract_default_feats()
            - _extract_time_of_day()
            - timediff
            - if microseconds are present:
                convert columns to timestamp

        Args:
            X (pd.DataFrame): data for datetime cols parsing.
            datetime_cols (list): datetime cols.
            created_datetime_cols (list): placeholder to fill with created
                datetime columns (for future reference).
            train (bool): Apply enclosed functions to train or test set.

        Returns:
            X (pd.DataFrame): data with parsed datetime cols.
            created_datetime_cols (list): list of created datetie cols.

        """
        for col in self._datetime_cols:
            try:

                if np.any(X[col].dt.microsecond):
                    # Divides and returns the integer value of the quotient. It dumps the digits after the decimal.
                    X[col] = X[col].values.astype(np.int64) // 10 ** 6
                else:
                    self._extract_default_feats(X, col, train, prefix=col+'_')
                    if col+'_hour' in X:
                        X[col+'_part_of_day'] = 0
                        X[col + '_part_of_day'] = X[col + '_part_of_day'].apply(self.extract_time_of_day)    
                        if train: # do not append when parsing test
                            self._created_datetime_cols.append(col+'_part_of_day')
            except:
                self.printer.print(f'DateParser._extract_default_feats error on {col}', order='error')

        if len(self._datetime_cols) == 2:
            try:
                # try because one of the two columns may be timestamp the other may be ordinary datetime
                timediff_colname = f'timediff_{self._datetime_cols[0]}_{self._datetime_cols[1]}'
                X[timediff_colname] = X[self._datetime_cols[0]] - X[self._datetime_cols[1]]
                X[timediff_colname] = X[timediff_colname] / np.timedelta64(1, 'D')
                if train: # do not append when parsing test
                    self._created_datetime_cols.append(timediff_colname)
            except:
                self.printer.print(f'DateParser timediff calculation error', order='error')

        if self.country:
            for col in self._datetime_cols:
                try:
                    X[f'{col}_holidays_flag'] = self.parse_holidays(X[col], self.country, self.state, self.prov, holiday_names = False)
                    X[f'{col}_holidays_name'] = self.parse_holidays(X[col], self.country, self.state, self.prov, holiday_names = True)
                    if train: # do not append when parsing test
                        self._created_datetime_cols.append(f'{col}_holidays_flag')
                        self._created_datetime_cols.append(f'{col}_holidays_name')
                except:
                    self.printer.print(f'DateParser._parse_holidays error on {col}', order='error')

        if self.payday:
            if f'{col}_day' in X:
                X[f'{col}_is_payday'] = X[f'{col}_day'].apply(self._get_payday_flag)
                if train: # do not append when parsing test
                    self._created_datetime_cols.append(f'{col}_is_payday')
                
        for col in self._datetime_cols:
            try:
                X[f'{col}_days_from_epoch'] = X[col].apply(self._get_days_from_epoch)
                if train: # do not append when parsing test
                    self._created_datetime_cols.append(f'{col}_days_from_epoch')
            except:
                self.printer.print(f'DateParser._get_days_from_epoch error on {col}', order='error')

        X.drop(self._datetime_cols, axis=1, inplace=True)
        return X

    def _exclude_false_datetime_cols(self, X):
        '''Find faulty datetime columns in the columns selected by dateutil.parser'''
        # Add methods below and to the methods list
        def coma_in_val(val):
            return ',' in val
        methods = [coma_in_val] 
        # ----------------------------
    
        # exclude real datetime cols from the following cross check
        proper_datetime_format_cols = X[self._datetime_cols].select_dtypes(include = ['datetime', 'datetime64', 'datetime64', 'datetime64[ns, UTC]']).columns.tolist()
    
        for col in self._datetime_cols:
            if col not in proper_datetime_format_cols:
                for method in methods:
                    if np.any(X[col].dropna().apply(coma_in_val)):
                        self._datetime_cols = [x for x in self._datetime_cols if x != col]
                    # if apply new method:
                    #     self._datetime_cols.remove(col)
        
    def _find_datetime_cols(self, X):
        """
        Find names of datetime cols in data.

        Use dateutil and a sample of 10 rows and all columns.
        Iterate over each value in each col and try to apply dateutil.parser.parse.
        Save cols names to self._datetime_cols list.

        Args:
            X (pd.DataFrame): raw data.

        Returns:
            None.

        """

        # this will be the default values for year/month/day if they are not present in string
        DEFAULT = datetime(2020, 1, 31)
        sample = X.sample(1000 if len(X) >= 1000 else(len(X)))
        datetime_cols = []
        for col in sample:
            #make sure that all the values in col are parsed as datetime
            try:            
                num_non_nan_vals_in_col = len(sample[col].dropna())
                num_parsed_dates_in_col = len(sample[col].dropna().apply(parser.parse, default = DEFAULT))                
                if (num_non_nan_vals_in_col > 0) & (num_non_nan_vals_in_col == num_parsed_dates_in_col):
                    datetime_cols.append(col)
            except:
                continue
        # grab the actual datetime cols that can come from xlsx metadata
        datetime_cols.extend(sample.select_dtypes(include = ['datetime', 'datetime64', 'datetime64', 'datetime64[ns, UTC]']).columns.tolist())
        self._datetime_cols = list(set(datetime_cols))
        self._exclude_false_datetime_cols(X)

    # -------------------------------------------------------------------------
    # quarter year format
    def _confirm_quarter_year_format(self, val):
        '''Confirm if string value can be transformed to pd.Timestamp'''
        transformed = self._transform_quarter_year_to_datetime_string(val)
        return type(pd.to_datetime(transformed)) == pd.Timestamp
    # .........................................................................
    def _transform_quarter_year_to_datetime_string(self, val):
        '''Convert quarter/year values to datetime-like string
            E.g. '1 Q 1990' -> '31/03/1990'
        '''        
        val_lst = val.split(' ')
        if len(val_lst) == 3:
            match_pattern = ' '.join(val_lst[:2])
            new_datetime_val = quarters[match_pattern]+'/'+val_lst[-1]
            return new_datetime_val
    # -------------------------------------------------------------------------
    # week year format
    def _extract_month_from_yearday(self, yearday):
        '''Extract month number integer from yearday integer.
            E.g. yearday 36 == month 2 & day 5. Extracted value = 2
        '''
        cnt = 0
        month = 0
        for months, days in month_days.items():
            while cnt < yearday:
                cnt+=days            
                month+=months
        month -= 1
        return month
    # .........................................................................    
    def _extract_monthday_from_yearday(self, yearday):
        '''Extract month day integer from yearday integer.
            E.g. yearday 36 == month 2 & day 5. Extracted value = 5
        '''
        yearday_copy = yearday
        for months, days in month_days.items():
            while yearday_copy > 0:
                yearday_copy -= days    
        day = month_days[months] - (abs(yearday_copy))
        return day
    # .........................................................................    
    def _extract_yearday_from_week_time_string(self, val):
        '''Extract year integer from week year string.
            E.g. '12 WK 1990' -> 1990
        '''
        val_lst = val.split(' ')
        week_val = ' '.join(val_lst[:2])
        yearday = week_yearday[week_val]
        return yearday
    # .........................................................................    
    def _transform_week_year_to_datetime_string(self, val):
        '''Convert week/year values to datetime-like string
            E.g. '12 WK 1990' -> '05/02/1990'
        '''
        val_lst = val.split(' ')
        if len(val_lst) == 3:
            yearday = self._extract_yearday_from_week_time_string(val)
            month = self._extract_month_from_yearday(yearday)
            monthday = self._extract_monthday_from_yearday(yearday)
    
            year_string = val_lst[-1]
            day_month_string = '/'.join([str(monthday), str(month)])
            new_datetime_val = '/'.join([day_month_string, year_string])
        return new_datetime_val
    # .........................................................................
    def _confirm_week_year_format(self, val):
        '''Confirm if string value can be transformed to pd.Timestamp'''
        new_datetime_val = self._transform_week_year_to_datetime_string(val)
        return type(pd.to_datetime((new_datetime_val))) == pd.Timestamp
    # .........................................................................
    def _find_transform_unconventional_cols(self, X):
        '''
        Find unconventional datetime columns E.g. '43 WK 1990' / '1 Q 2005', 
        transform into datetime-like string

        Parameters
        ----------
        X : pd.DataFrame
            data for examination/transformation.

        Returns
        -------
        X : pd.DataFrame
            data transformed.

        '''
        sample = X.sample(1000 if len(X) >= 1000 else(len(X)))
        for col in sample.select_dtypes(include = 'O'):
            # quarter based cols transformation
            try:
                if np.all(sample[col].apply(self._confirm_quarter_year_format)):
                    X[col] = X[col].apply(self._transform_quarter_year_to_datetime_string)
            except:
                pass
            # week based cols transformation
            try:
                if np.all(sample[col].apply(self._confirm_week_year_format)):
                    X[col] = X[col].apply(self._transform_week_year_to_datetime_string)
            except:
                pass
        return X
    # -------------------------------------------------------------------------
    # executed separately - transform cols with month val E.g. 'JAN', 'FEB' into integers
    def _confirm_month_val(self, val):
        '''Confirm if value is in months dictionary'''
        return val in months.keys()
    # .........................................................................    
    def _find_transform_month_string_to_integer(self, X, train = True):
        '''
        Find/transform columns containing month codes ('JAN', 'FEB') into integers.

        Parameters
        ----------
        X : pd.DataFrame
            data for searching/transforming month codes.
        train : bool, optional
            train/test set transformation flag. If True: save the transformed 
            column name to class instance. 
            The default is True.

        Returns
        -------
        X : pd.DataFrame
            transformed data.

        '''
        sample = X.sample(1000 if len(X) >= 1000 else(len(X)))

        for col in sample.select_dtypes(include = 'O'):
            try:
                if np.all(sample[col].apply(self._confirm_month_val)):
                    X[f'{col}_month_number'] = X[col].map(months)
                    X.drop(col, axis = 1, inplace = True)
                    if train:
                        self._created_datetime_cols.append(f'{col}_month_number')
                    if self.verbose:
                        print('   - Found columns with month codes (E.g. "Jan", "Feb")/transformed into integers.')
            except:
                continue
        return X
    # -------------------------------------------------------------------------
    # determine dayfirst argument
    def _get_date_delimiter(self, date):
        '''Find delimiter from constant DEFAULT_DELIMITERA with max occurances in date string.'''
        heat_dict = {i: len(date.split(i)) for i in DATE_DELIMITERS}
        return max(heat_dict, key=heat_dict.get)
    # .........................................................................        
    def _is_numeric_date(self, date):
        '''Check if date can be split by delimiter in multiple components'''
        delimiter = self._get_date_delimiter(date)
        num_date_components = len(date.split(delimiter))
        if num_date_components > 1:
            return True
        else:
            return False
    # .........................................................................            
    def _need_determine_dayfirst_argument(self, datetime_col_series):
        '''Check if need dayfirst afgument for pd.to_datetime based on date format'''
        sample_date = datetime_col_series.dropna()[0]
        return self._is_numeric_date(sample_date)
    # .........................................................................            
    def _is_dayfirst(self, datetime_col_series):
        '''Determine dayfirst argument for pd.to_datetime based on maximum of date components'''
        clean_series = datetime_col_series.dropna()
        delimiter = self._get_date_delimiter(clean_series[0])
        date_components = clean_series.str.split(' ', expand = True)[0]
        date_components = date_components.str.split(delimiter, expand = True)
        max_values_in_date_components = {}
        for col in date_components:
            date_components[col] = date_components[col].astype(int)
            max_values_in_date_components[col] = date_components[col].max()
        if 12 < max_values_in_date_components[0] <= 31:
            dayfirst_arg = True
        else:
            dayfirst_arg = False
        return dayfirst_arg
    # =============================================================================
    def fit_transform(self, df):
        """
        Parse datetime cols in data and create derivative features.

        Possible new features: [year, month, day, quarter, week, weekday, dayofyear,
                                hour, minute, second, part_of_day, timediff].
        If df contains 2 datetime columns, timediff column is created showing
            the difference between two datetime columns in days
        If df contains > 1 datetime columns, features are parsed from all of them
            with corresponding prefixes in new features names.
        Original datetime column(s) is deleted from df.

        Args:
            df (pd.DataFrame): raw data.
        Returns:
            df (pd.DataFrame): data with new features parsed from datetime columns.

        """
        self.printer.print('Parsing dates', order=1)
        try:
            # try parsing month strings to month numbers on original data
            # first unconditional transformation
            df_copy = df.copy()
            df_copy = self._find_transform_month_string_to_integer(df_copy, train = True)
            # -----------------------------------------------------------------
            X = df_copy.copy()
            X = self._find_transform_unconventional_cols(X)
            self._find_datetime_cols(X)
            if self._datetime_cols:
                # convert to datetime
                for col in self._datetime_cols:
    
                    if self._need_determine_dayfirst_argument(X[col]):
                        dayfirst_arg = self._is_dayfirst(X[col])
                    else:
                        dayfirst_arg = False

                    X[col] = pd.to_datetime(X[col], 
                                            dayfirst = dayfirst_arg, 
                                            errors='coerce', 
                                            infer_datetime_format = True) # errors = 'coerse' allows to parse columns even if any value in datetime column is a mistake. coerce will set it to NaT.
                    # for misclassified potential datetime cols check if converted to datetime as NaN
                    if np.all(X[col].isnull()):
                        # convert back the unlucky pd.to_datetime attempt to original format
                        X[col] = df[col]
                        self._datetime_cols = [x for x in self._datetime_cols if x != col]
                #if len(self._datetime_cols) == 1:
                X = self._extract_all_feats(X)
                self.printer.print(f'Found and processed {len(self._datetime_cols)} date related columns', order=3)
                self.printer.print(f'Created {len(self._created_datetime_cols)} new date related features', order=3)
                if len(self._datetime_cols) == 2:
                    self.printer.print('Introduced date/time difference feature', order=3)
                gc.collect()
                return X
            else:               
                self.printer.print('No datetime cols found', order=2)
                return df_copy
        except:
            self._datetime_cols = None
            self._created_datetime_cols = None
            self.printer.print(f'Parse dates error', order='error')
            return df_copy

    def _align_test_columns_after_transform(self, X, original_test_cols):
        '''
        Align test set columns after transformation:
            - add blank (np.zeros) columns if not extracted out of test set datetime cols
            - remove the extra column that had been extracted from the test set datetime cols

        Parameters
        ----------
        X : pd.DataFrame
            data after extracting datetime cols.
        original_test_cols : list of strings
            original test set columns before extracting datetime features.

        Returns
        -------
        X : pd.DataFrame
            dataset with aligned columns.

        '''
        new_datetime_cols_in_test = [col for col in X if col not in original_test_cols]
        to_drop_from_test = [col for col in new_datetime_cols_in_test if col not in self._created_datetime_cols]
        to_add_to_test = [col for col in self._created_datetime_cols if col not in new_datetime_cols_in_test]
        X.drop(to_drop_from_test, axis=1, inplace=True)
        for col in to_add_to_test:
            X[col] = 0
        return X

    def transform(self, df):
        """
        Parse datetime cols from test set using the fitted class instance.

        Use the saved instance agrument self._datetime_cols to parse the defined cols.
        Align the parsed cols with the instance argument self._created_datetime_cols

        Args:
            X (pd.DataFrame): raw data.

        Returns:
            X (pd.DataFrame): data with new features parsed from datetime columns.

        """

        original_test_cols = df.columns.tolist()
        df_copy = df.copy()

        if self._datetime_cols:
            df_copy = self._find_transform_month_string_to_integer(df_copy, train = False)
            X = df_copy.copy()
            X = self._find_transform_unconventional_cols(X)
            # convert to datetime
            for col in self._datetime_cols:
                if col in X:
                    X[col] = pd.to_datetime(X[col], errors='coerce')
            X = self._extract_all_feats(X, train=False)
            self.printer.print(f'Found and processed {len(self._datetime_cols)} date related columns', order=3)
            self.printer.print(f'Created {len(self._created_datetime_cols)} new date related features', order=3)
            if len(self._datetime_cols) == 2:
                self.printer.print('Introduced date/time difference feature', order=4)
            X = self._align_test_columns_after_transform(X, original_test_cols)
            return X
        else:
            self.printer.print('No datetime cols found', order=2)
            #df_copy = self._align_test_columns_after_transform(df_copy, original_test_cols)
            return df_copy
