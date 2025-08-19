from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from data_loader import telco_data_loader
from data_processor import processTelcoData

# Setting up a logger
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

logger = logging.getLogger(__name__)

# Importing the data
raw_data = telco_data_loader()

# cleaning the data
data_for_fitting = processTelcoData(raw_data)

# Creating a pipeline
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=1000))]
)

# creating a cv engine
cv = StratifiedKFold(n_splits=5, shuffle=True)

# Creating a dict of scoring
scoring = {"accuracy": "accuracy"}

# running the pipeline
cv_results = cross_validate(
    pipeline,
    X=data_for_fitting["predictors"],
    y=data_for_fitting["target"],
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1,
)

# logging the results
logger.info(cv_results["test_accuracy"])
