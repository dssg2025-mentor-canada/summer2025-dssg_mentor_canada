# how does having mentor present effect odds of pursuing further education for low SES individuals?
# --- Imports ---
import pandas as pd
import statsmodels.api as sm

# --- Load Data ---
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')

# --- Counts ---
ses_indicators = ['38_food_bank_use','38_social_assistance','38_work_to_support_family']

ses_data = data[(
        (data['38_food_bank_use'] == 1) |
        (data['38_social_assistance'] == 1) |
        (data['38_work_to_support_family'] == 1)) |
        ((data['38_food_bank_use'] == 2) &
        (data['38_social_assistance'] == 2) &
        (data['38_work_to_support_family'] == 2))].copy()

ses_data[ses_indicators] = ses_data[ses_indicators].replace({2: 0})
ses_data['low_ses'] = ses_data[ses_indicators].any(axis=1).astype(int)

print('Further education value counts', data['13_further_edu'].value_counts())

# cross table (use ses_data so we can for sure state socioecon status)
table = pd.crosstab(ses_data['low_ses'], ses_data['13_further_edu'])
print(table)

# --- Filter and Clean ---
logistic_data = ses_data.copy()
logistic_data['13_further_edu'] = logistic_data['13_further_edu'].replace({'Yes': 1,
                                         'No': 0})

print(logistic_data['13_further_edu'])

# --- Logistic Model ---

# Reweight
target = logistic_data['13_further_edu'].astype(int)


n_total = len(target)
n_pos = (target == 1).sum()  
n_neg = (target == 0).sum()  
w_pos = n_total / (2 * n_pos)  
w_neg = n_total / (2 * n_neg) 

X = logistic_data['low_ses']
X = sm.add_constant(X)  # Add intercept

logistic_data['asst_weights'] = target.apply(lambda x: w_pos if x == 1 else w_neg)

log_reg = sm.GLM(target, X, family=sm.families.Binomial(), freq_weights=logistic_data['asst_weights']).fit()

print(log_reg.summary())


