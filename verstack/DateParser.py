import pandas as pd
import numpy as np
import pprint
from verstack.tools import Printer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class DateParser:

    __version__ = "0.1.2"

    def __init__(self, verbose=True):
        """
        Initialize DateParser instance.

        Automatically finds datetime columns, converts them to datetime objects, extracts
        datetime features and removes original datetime columns.

        Parameters
        ----------
        verbose : bool, optional
            Print progress to stdout. The default is True.

        Returns
        -------
        None.

        """
        self._datetime_cols = None
        self._created_datetime_cols = (
            {}
        )  # save created datetime columns for transform method
        self.dayfirst = (
            {}
        )  # save dayfirst parameter for each datetime column for transform method
        self.verbose = verbose
        self.printer = Printer(verbose=verbose)

    def __repr__(self):
        return f"DateParser(datetime_cols={self._datetime_cols},\
            \n           created_datetime_cols: {self._created_datetime_cols}\
            \n           verbose: {self.verbose}"

    @property
    def datetime_cols(self):
        return self._datetime_cols

    @property
    def created_datetime_cols(self):
        return self._created_datetime_cols

    # verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, bool):
            print(
                f"{value} is not a valid verbose argument, must be a bool, setting to True"
            )
            self._verbose = True
        else:
            self._verbose = value

    def col_contains_dates(self, series):
        """Check if column contains date-like strings"""
        return all(
            series.str.contains(r"\d{1,4}.\d{1,2}.\d{1,4}", regex=True, na=True)
        )

    def find_datetime_cols(self, df):
        """Find all columns that contain date-like strings and convert to datetime objects"""
        datetime_cols = []
        for col in df.select_dtypes(include="object"):
            if self.col_contains_dates(df[col]):
                converted_to_datetime = self.convert_to_datetime(df[col])
                if converted_to_datetime.dtype == "datetime64[ns]":
                    df[col] = converted_to_datetime
                    datetime_cols.append(col)
        if len(datetime_cols) > 0:
            self._datetime_cols = datetime_cols
        if self._datetime_cols is not None:
            self.printer.print(
                f"Datetime columns found: {self._datetime_cols}", order=2
            )
        return df

    def is_year_four_digits(self, dt_str_series):
        """Check if year is four digits"""
        return all(dt_str_series.str.contains(r"\d{4}", regex=True, na=True))

    def infer_dayfirst_argument(self, dt_str_series):
        """Infer dayfirst argument for pd.to_datetime based on date-like strings

        If dayfirst=True returns more null values than dayfirst=False, return False.
        If dayfirst=False returns more null values than dayfirst=True, return True.
        If both methods return the same number of null values, return None.

        """
        dayfirst = True
        null_vals_before_parse = dt_str_series.isnull().sum()
        dt_series_parsed = pd.to_datetime(
            dt_str_series, dayfirst=dayfirst, errors="coerce"
        )
        null_vals_after_parse = dt_series_parsed.isnull().sum()
        if null_vals_after_parse > null_vals_before_parse:
            dayfirst = False
        return dayfirst

    def convert_to_datetime(self, dt_str_series):
        """Convert date-like columns to datetime objects

        First try to parse with dayfirst=True, then try with dayfirst=False,
        If both methods return more null values than before parsing, return original series.

        """
        if self.is_year_four_digits(dt_str_series):
            dayfirst = None
            dt_series_parsed = pd.to_datetime(dt_str_series, errors="coerce")
            self.dayfirst[dt_str_series.name] = dayfirst
            return dt_series_parsed
        else:
            dayfirst = self.infer_dayfirst_argument(dt_str_series)
            dt_series_parsed = pd.to_datetime(
                dt_str_series, dayfirst=dayfirst, errors="coerce"
            )
            self.dayfirst[dt_str_series.name] = dayfirst
            return dt_series_parsed

    def extract_date_feature(self, series, feature, fit_transform=True):
        """Extract date feature from datetime columns

        If feature has more than one unique value, return the feature, else return None.
        If fit_transform is False, return the feature regardless of number of unique values
        (applicable for transform method) because regardless of number of unique values,
        this feature had been extracted at fit_transform.

        Parameters
        ----------
        series : pandas.Series
            Series containing datetime objects
        feature : str
            Date feature to extract
        fit_transform : bool, default=True
            If True, return None if feature has only one unique value, else return the feature.
            If False, return the feature regardless of number of unique values.

        Returns
        -------
        pandas.Series
            Series containing extracted date feature or None

        """
        extracted_feature = eval(f"series.dt.{feature}")
        if not fit_transform:
            return extracted_feature
        if extracted_feature.nunique() > 1:
            return extracted_feature
        else:
            return None

    def extract_week(self, series, fit_transform=True):
        """Extract week from datetime columns

        If week has more than one unique value, return the week, else return None.
        If fit_transform is False, return the week regardless of number of unique values
        (applicable for transform method) because regardless of number of unique values,
        this feature had been extracted at fit_transform.

        Parameters
        ----------
        series : pandas.Series
            Series containing datetime objects
        fit_transform : bool, default=True
            If True, return None if week has only one unique value, else return the week.
            If False, return the week regardless of number of unique values.

        Returns
        -------
        pandas.Series
            Series containing extracted week or None

        """
        try:
            extracted_week = series.apply(lambda x: x.isocalendar()[1])
            week = extracted_week
        except ValueError:
            extracted_week = series.dropna().apply(lambda x: x.isocalendar()[1])
            week = pd.Series(index=series.index)
            week.loc[extracted_week.index] = extracted_week
        if not fit_transform:
            return week
        if week.nunique() > 1:
            return week
        else:
            return None

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

    def extract_date_features(self, df, col):
        """Extract date features from datetime columns

        If feature has more than one unique value, create a new column with the feature,
        else do not create a new column.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to extract date features from
        col : str
            Datetime column to extract date features from

        Returns
        -------
        pandas.DataFrame
            DataFrame with date features extracted from datetime column

        """
        features = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekday",
            "quarter",
        ]
        for feature in features:
            extracted_feature = self.extract_date_feature(df[col], feature)
            if extracted_feature is not None:
                df[col + "_" + feature] = extracted_feature
                if self._created_datetime_cols.get(col) is None:
                    self._created_datetime_cols[col] = []
                self._created_datetime_cols[col].append(feature)
        week = self.extract_week(df[col])
        if week is not None:
            df[col + "_week"] = week
            if self._created_datetime_cols.get(col) is None:
                self._created_datetime_cols[col] = []
            self._created_datetime_cols[col].append("week")
        if "hour" in self._created_datetime_cols[col]:
            df[f"{col}_part_of_day"] = df[f"{col}_hour"].apply(
                self.extract_time_of_day
            )
            self._created_datetime_cols[col].append("part_of_day")
        return df

    def fit_transform(self, df):
        """Find datetime columns and extract date features from them

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to look for datetime columns and extract date features from

        Returns
        -------
        pandas.DataFrame
            DataFrame with datetime columns converted to datetime objects and
            date features extracted from them

        """
        self.printer.print("Looking for datetime columns", order=1)
        try:
            data = df.copy()
            self.find_datetime_cols(data)
            if self._datetime_cols is None:
                self.printer.print("No datetime columns found", order=2)
                return df
            for col in self._datetime_cols:
                data = self.extract_date_features(data, col)
                data.drop(col, axis=1, inplace=True)
            self.printer.print(
                "Extracted following datetime features:", order=2
            )
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self._created_datetime_cols)
        except Exception as e:
            self.printer.print(
                "Error at DateParser.fit_transform. Returning untransformed df",
                order="error",
            )
            print(e)
            self._datetime_cols = None
            return df
        return data

    def transform(self, df):
        """Extract date features from datetime columns

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to extract date features from based on fit_transform artefacts

        Returns
        -------
        pandas.DataFrame
            DataFrame with date features extracted from datetime columns

        """
        if self._datetime_cols is None:
            self.printer.print(
                "No datetime columns were found at fit_transform", order=1
            )
            return df
        for col in self._datetime_cols:
            if self.dayfirst.get(col) is None:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = pd.to_datetime(
                    df[col], dayfirst=self.dayfirst[col], errors="coerce"
                )
            for feature in self._created_datetime_cols[col]:
                if feature not in ["week", "part_of_day"]:
                    df[f"{col}_{feature}"] = self.extract_date_feature(
                        df[col], feature, fit_transform=False
                    )
            if "week" in self._created_datetime_cols[col]:
                df[f"{col}_week"] = self.extract_week(
                    df[col], fit_transform=False
                )
            if "hour" in self._created_datetime_cols[col]:
                df[f"{col}_part_of_day"] = df[f"{col}_hour"].apply(
                    self.extract_time_of_day
                )
            df.drop(col, axis=1, inplace=True)
        self.printer.print("Extracted following datetime features:", order=2)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self._created_datetime_cols)
        return df
