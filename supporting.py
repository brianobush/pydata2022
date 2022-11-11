
# For data engineering node

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    return train_df, test_df




# For data science node:

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def train_model(train_df: pd.DataFrame, contamination_factor: float):
    # Note: Could parameterize n_estimators too
    clf = IsolationForest(random_state=69, bootstrap=True, contamination=contamination_factor)
    clf.fit(train_df.iloc[:,1:]) 
    print(train_df.iloc[:,1:])
    return clf

def predict(ml_model, test_df: pd.DataFrame):
    test_df['anomaly_score'] = ml_model.score_samples(test_df.iloc[:,1:]) 
    return test_df


# Data Engineering pipeline

from .nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="temperature_database",
            outputs=["train_data", "test_data"],
            name="node_train_test_split"
            ),

    ])

# Data Science pipeline

from .nodes import train_model, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["train_data", "params:contamination_factor"],
            outputs="ml_model",
            name="node_train_model"
            ),
        node(
            func=predict,
            inputs=["ml_model", "test_data"],
            outputs="predictions",
            name="node_predict"
            ),

    ])

# Pipeline registry



from pydata2022.pipelines import (
    data_engineering as de,
    data_science as ds
)

def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline        
    }


