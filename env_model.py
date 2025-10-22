import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,accuracy_score, classification_report

df = pd.read_csv('air_quality_dataset.csv')

df['DATEOFF'] = pd.to_datetime(df['DATEOFF'], errors='coerce')
df['DATEON'] = pd.to_datetime(df['DATEON'], errors='coerce')

numeric_columns = df.columns.drop(['SITE_ID', 'DATEOFF', 'DATEON']).tolist()

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

#https://www.kaggle.com/code/atifmasih/air-qaulity-categorization-using-randomforest-94
