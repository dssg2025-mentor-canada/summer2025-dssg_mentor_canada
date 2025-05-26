import pandas as pd
import os
from siuba import _, group_by, summarize, filter, select, mutate, arrange, count

youth = pd.read_csv('../../dssg-2025-mentor-canada/Data/intermediate.csv')

### Reversing one-hot encoding
gender_cols = youth.loc[:,'QS1_9_GENDER1_1_1':'QS1_9_GENDER1_6_6']
youth['QS1_9_gender'] = gender_cols.idxmax(axis = 1)
youth['QS1_9_gender'].head().reset_index()
