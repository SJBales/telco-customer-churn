from sklearn.linear_model import LogisticRegression
import logging
from data_loader import telco_data_loader
from data_processor import processTelcoData

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

# Importing the data
raw_data = telco_data_loader()

# cleaning the data
data_for_fitting = processTelcoData(raw_data)

# Training the model
regressor = LogisticRegression()

fitted_model = regressor.fit(data_for_fitting["predictors"], data_for_fitting["target"])
