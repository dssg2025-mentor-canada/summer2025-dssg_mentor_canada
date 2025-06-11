import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("../../dssg-2025-mentor-canada/Data/youth_tidy.csv")

predictor_cols = ["38_social_assistance",
                  "38_work_to_support_family",
                  "38_food_bank_use"]
target_cols = ["18d_early_mentor_unmet_access"]

print('count of those with guardians that required social assistance during youth (ages 12–18):', (df['38_social_assistance'] == 1).sum())
print('count of those who needed to work to support their families during youth (ages 12-18):', (df['38_work_to_support_family'] == 1).sum())
print('count of those who used food banks during youth (ages 12-18):', (df['38_food_bank_use'] == 1).sum())
print("count of those who wanted a mentor but had unmet access during childhood (ages 6-11):",
      (df['18d_early_mentor_unmet_access']==1).sum())
print("\nRaw target class distributions:")

print(df['18d_early_mentor_unmet_access'].value_counts())

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

# check for class imbalance -> if so, reweight
logistic_data['18d_early_mentor_unmet_access'].value_counts()

X = logistic_data[predictor_cols].astype('float64')
X = sm.add_constant(X)
y = logistic_data['18d_early_mentor_unmet_access'].astype('float64')

model = sm.Logit(y, X).fit()

print(model.summary())
