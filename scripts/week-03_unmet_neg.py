import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("../../dssg-2025-mentor-canada/Data/youth_tidy.csv")

predictor_cols = ["38_social_assistance", "38_work_to_support_family", "38_food_bank_use"]

# recode 18d_early_mentor_unmet_access to be binary
logistic_data = df[['18d_early_mentor_unmet_access',
                    "38_social_assistance", 
                    "38_work_to_support_family", 
                    "38_food_bank_use"]].copy()

logistic_data['18d_early_mentor_unmet_access'] = (
    logistic_data['18d_early_mentor_unmet_access']
    .map({'Yes': 1, 'No': 0})
    .astype('Int64')
)

logistic_data['38_social_assistance'] = (
    logistic_data['38_social_assistance']
    .map({2: 0})
    .astype('Int64')
)

logistic_data['38_work_to_support_family'] = (
    logistic_data['38_work_to_support_family']
    .map({2: 0})
    .astype('Int64')
)

logistic_data['38_food_bank_use'] = (
    logistic_data['38_food_bank_use']
    .map({2: 0})
    .astype('Int64')
)


# keep only observations that had yes or no answers
valid_rows = logistic_data[[
    "38_social_assistance", 
    "38_work_to_support_family", 
    "38_food_bank_use",
    "18d_early_mentor_unmet_access"
]].isin([1, 0]).all(axis=1)


# filter data
logistic_data = logistic_data[valid_rows]

# # keep only observations that had yes or no answers
logistic_data = df['18d_early_mentor_unmet_access'].isin([1,2])

X = logistic_data[predictor_cols]
X = sm.add_constant(X)
y = logistic_data['18d_early_mentor_unmet_access'].astype(float)

model = sm.Logit(y, X).fit()

print(model.summary())