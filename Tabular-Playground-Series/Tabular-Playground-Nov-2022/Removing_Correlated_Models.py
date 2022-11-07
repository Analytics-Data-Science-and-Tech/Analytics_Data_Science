import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

## Reading data-files
df = pd.read_parquet('s3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/preds_logit_concat_gzip.parquet', engine = 'fastparquet')

## train and test
# preds_df = df.clip(0, 1) 
# train = preds_df[preds_df['target'].notnull()]
# test = preds_df[preds_df['target'].isnull()] 
