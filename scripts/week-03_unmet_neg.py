import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("../../dssg-2025-mentor-canada/Data/youth_tidy.csv")

predictor_cols = ["38_social_assistance",
                  "38_work_to_support_family",
                  "38_food_bank_use"]
target_cols = ["18d_early_mentor_unmet_access"]

# # recode 18d_early_mentor_unmet_access to be binary
# logistic_data = df[
#     df["38_social_assistance"].isin([1, 2]) &
#     df["38_work_to_support_family"].isin([1, 2]) &
#     df["38_food_bank_use"].isin([1, 2])
# ].copy()

logistic_data = df.copy()

logistic_data['18d_early_mentor_unmet_access'] = (
    logistic_data['18d_early_mentor_unmet_access']
    .map({'Yes': 1, 'No': 2})
    .astype('Int64')
)

# recode 18d_early_mentor_unmet_access to be binary
logistic_data = logistic_data[
    logistic_data["38_social_assistance"].isin([1, 2]) &
    logistic_data["38_work_to_support_family"].isin([1, 2]) &
    logistic_data["38_food_bank_use"].isin([1, 2]) &
    logistic_data["18d_early_mentor_unmet_access"].isin([1, 2])
].copy()

logistic_data[predictor_cols] = (
    logistic_data[predictor_cols]
    .replace({2: 0})
    .astype('Int64')
)

logistic_data[target_cols] = (
    logistic_data[target_cols]
    .replace({2: 0})
    .astype('Int64')
)

# logistic_data['38_food_bank_use'] = (
#     logistic_data['38_food_bank_use']
#     .map({2: 0})
#     .astype('Int64')
# )


# # keep only observations that had yes or no answers
# valid_rows = logistic_data[[
#     "38_social_assistance", 
#     "38_work_to_support_family", 
#     "38_food_bank_use",
#     "18d_early_mentor_unmet_access"
# ]].isin([1, 0]).all(axis=1)


# # filter data
# logistic_data = logistic_data[valid_rows]

# # # # keep only observations that had yes or no answers
# # logistic_data = df['18d_early_mentor_unmet_access'].isin([1,2])

X = logistic_data[predictor_cols].astype('float64')
X = sm.add_constant(X)
y = logistic_data['18d_early_mentor_unmet_access'].astype('float64')

model = sm.Logit(y, X).fit()

print(model.summary())

# print(X.dtypes)
# print(y.dtypes)

# print(logistic_data[["38_social_assistance",
#                   "38_work_to_support_family",
#                   "38_food_bank_use",
#                   "18d_early_mentor_unmet_access"]])