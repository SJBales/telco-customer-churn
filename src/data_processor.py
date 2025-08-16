import pandas as pd
import logging
from typing import Tuple, Dict, Union

# Configuring the logger
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

logger = logging.getLogger(__name__)


class telcoDataCleaner:

    def __init__(self):
        self.numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        self.Target_column = "Churn"

    # creating a master method to run all the sub-cleaning functions
    def clean_data(
        self, df: pd.DataFrame
    ) -> Dict[str, Union[pd.DataFrame | pd.Series]]:

        clean_df = df.copy()

        # creating a column for churn indicator
        clean_df = self._standardize_categories_(clean_df)

        # converting datatypes
        clean_df = self._convert_data_types_(clean_df)

        # doing an outlier summary
        clean_df = self._detect_outliers_(clean_df)

        # cleaning missing values
        clean_df = self._identify_missing_values_(clean_df)

        # filling missing values
        clean_df = self._fill_missing_values_(clean_df, ["TotalCharges", "Churn"])

        # Prepping predictors and target column
        preds, target = self._prep_data_(clean_df)

        return {"clean_table": clean_df, "predictors": preds, "target": target}

    # Converting numeric data types
    def _convert_data_types_(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Data types successfully converted")

        return df

    # Creating a method to standardize binary columns
    def _standardize_categories_(self, df: pd.DataFrame) -> pd.DataFrame:

        # Flagging binary columns
        binary_columns = [
            "Partner",
            "Dependents",
            "PhoneService",
            "Churn",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "OnlineSecurity",
        ]
        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})

        logger.info("Binary columns converted")

        # Converting genders to binaries
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
        logger.info("Converting gender")

        return df

    # Detecting outliers
    def _detect_outliers_(self, df: pd.DataFrame) -> pd.DataFrame:

        outlier_summary = {}

        for col in self.numeric_columns:
            if (col in df.columns) & (col != self.Target_column):
                lq = df[col].quantile(q=0.25)
                uq = df[col].quantile(q=0.75)
                iqr = uq - lq

                lb = lq - (1.5 * iqr)
                ub = uq + (1.5 * iqr)

                outliers = df[(df[col] < lb) | (df[col] > ub)]
                outlier_count = len(outliers)

                if outlier_count > 0:
                    outlier_summary[col] = outlier_count

        logger.info(f"Outlier Summary: {outlier_summary}")

        return df

    # Handling missing values
    def _identify_missing_values_(self, df: pd.DataFrame) -> pd.DataFrame:

        missing_value_summary = {}

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_value_summary[col] = df[col].isnull().sum()

        logger.info(f"Columns with missing values: {missing_value_summary}")

        return df

    # Drop or fill missing values
    def _fill_missing_values_(self, df: pd.DataFrame, cols) -> pd.DataFrame:

        for col in cols:
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"Filled missing values in {col} with medians")

        return df

    # Prepping the table for fitting
    def _prep_data_(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:

        predictor_cols = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "TotalCharges",
            "MonthlyCharges",
        ]
        predictors = df.loc[:, predictor_cols]

        return predictors, df.loc[:, self.Target_column]


def processTelcoData(raw_df):

    processor = telcoDataCleaner()

    processed_data = processor.clean_data(raw_df)

    logger.info("Data has been cleaned")

    return processed_data


if __name__ == "__main__":

    from data_loader import telco_data_loader

    raw_data = telco_data_loader()

    processed_data = processTelcoData(raw_data)
