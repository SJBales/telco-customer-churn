import logging
import pandas as pd
import kagglehub
from pathlib import Path

# Configuring the logger
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

logger = logging.getLogger(__name__)


# Defining the loader function
def telco_data_loader(cache_dir: str = "data/raw") -> pd.DataFrame:
    """
    Downloads the telco data from kagglehub

    Args
        and returns a pandas dataframe

    Returns
        Dataframe of telco customer churn data
    """

    try:
        logger.info("Starting download of telco data")

        # Dowloading the data and setting the path
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        logger.info("Telco data download successful")

        # Creating the path to the CSV file
        csv_file = Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

        # Reading the CSV file
        df = pd.read_csv(csv_file)
        logger.info("CSV data downloaded successfully")

        # Returning the dataframe if it works
        return df

    except Exception as e:
        logger.error(f"Failed to load data {e}")
        raise


if __name__ == "__main__":
    df = telco_data_loader()
