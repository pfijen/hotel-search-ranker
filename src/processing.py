# %%
from typing import List, Tuple

import numpy as np
import pandas as pd


class Processor(object):
    """Class with all parsing and clearning related functions"""

    def __init__(self, logger) -> None:
        self.logger = logger
        pass

    def read_data(self, path: str) -> pd.DataFrame:
        self.logger.info("Loading the data")
        df = pd.read_csv(path, sep=",")
        return df

    def save_data(self, df, path: str, nrows: int) -> None:
        self.logger.info(f"Writing data sample of size {nrows} to {path}")
        df[:nrows].to_csv(path, index=False)
        pass

    def process(self, df) -> pd.DataFrame:
        """DataFrame processing pipeline of basic operations"""
        self.logger.info("Processing the data")
        df_processed = (
            df
            .pipe(self._parse_dates)
            .pipe(self._drop_cols_logic)
            .pipe(self._drop_cols_comp)
            .pipe(self._fill_zeros)
        )
        self.logger.info("Processing finished")
        return df_processed

    def split_data(
        self, df, train_size: float = 0.7, val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create datasets for train, val, test
        Full dataset contains 200k searches,
        len(df.srch_id.unique()) -> 199,795.
        Searches are on chronological order so split can be done
        by cutting off based on srch_id.

        Args:
            df ([pd.DataFrame]): Full dataset unsplitted

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: [
                Train set containing first 70% (if default),
                Validation set containing next 20% (if default),
                Test set containing remaining data (10% default)
            ]
        """
        self.logger.info("Splitting into train, val, test")

        split1, split2 = self._calculate_split_ids(df, train_size, val_size)

        df_train = df[df["srch_id"] <= split1]
        df_val = df[(df["srch_id"] > split1) & (df["srch_id"] <= split2)]
        df_test = df[df["srch_id"] > split2]

        self.logger.info("length of train set: " + str(len(df_train)))
        self.logger.info("length of validation set: " + str(len(df_val)))
        self.logger.info("length of test set: " + str(len(df_test)))

        return df_train, df_val, df_test

    def split_Xy(self, df) -> Tuple[pd.DataFrame, pd.Series]:
        """Split the dataframe into X and y
        In the original kaggle competition the bookings had
        a weight of 4 and clicks of 1 for the ndcg_at_k metric
        These weights are handled in the xgb model
        by setting the label_gain
        """
        self.logger.info("Splitting DataFrame into X, y")
        y = df["srch_id"].to_frame()
        y = y.assign(r=df["click_bool"].add(df["booking_bool"]))
        y = y.drop(columns=["srch_id"])["r"]

        X = df.drop(columns=["click_bool", "booking_bool"])
        return X, y

    def numeric_and_complete_cols(self, df) -> List[str]:
        """create a list of columns which are complete and numeric"""
        missing_values = df.isna().sum()
        no_missing = missing_values[missing_values == 0]
        col_list = [
            x
            for x in no_missing.index
            if not x.endswith("id") and not x.endswith("bool")
        ]
        return col_list

    def _parse_dates(self, df) -> pd.DataFrame:
        """Transforms string to datetime object
        Followed by calculating the day of week,
        month and quarter for seasonal features
        """
        self.logger.info("\tparsing the date_time")
        df_dt = df.assign(date_time=pd.to_datetime(df["date_time"]))
        df_parsed = df_dt.assign(
            day_of_week=df_dt["date_time"].dt.dayofweek + 1,
            month=df_dt["date_time"].dt.month,
        )
        return df_parsed

    def _drop_cols_logic(self, df) -> pd.DataFrame:
        """The competition belonging to this data
        has a formal test set which does not contain
        the position or gross booking.
        Kaggle Expedia reccommender competition.
        """
        self.logger.info("\tdropping columns not usable")

        drop_list = ["date_time", "position", "gross_bookings_usd"]
        df_dropped = df.drop(columns=drop_list)
        return df_dropped

    def _drop_cols_comp(self, df) -> pd.DataFrame:
        """The competitor columns have no added value
        in predictive performance and are half of the
        columns

        The second set of columns is dropped to
        reduce dimensionality without a major
        loss in performance. This saves training
        time and makes shap faster.
        """
        self.logger.info("\tdropping columns competitors")

        for col in df.columns:
            if col.startswith("comp"):
                df = df.drop(columns=col)
        return df

    def _fill_zeros(self, df) -> pd.DataFrame:
        """By defenition the zeros represent missing values in this dataset
        This functions replaces the zeros with np.nan
        """
        self.logger.info("\tFilling zeros with NaNs")

        fill_names = [
            "prop_starrating",
            "prop_review_score",
            "prop_log_historical_price",
        ]
        for col_name in fill_names:
            df[col_name] = df[col_name].replace(to_replace=0, value=np.nan)
        return df

    def _calculate_split_ids(
        self, df, train_size: float, val_size: float
    ) -> Tuple[int, int]:
        srches = df["srch_id"].unique()
        split1 = srches[round(len(srches) * train_size)]
        split2 = srches[round(len(srches) * (train_size + val_size))]
        return split1, split2
