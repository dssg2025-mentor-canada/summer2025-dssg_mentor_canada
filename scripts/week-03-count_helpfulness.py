# --- Imports ---
import pandas as pd
import statsmodels.api as sm

# --- Load Data ---
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')

# --- Counts ---
data['32_mentor_helpfulness'] = data['32_mentor_helpfulness'].replace({1.0: 'Very helpful',
                                       2.0: 'Fairly helpful',
                                       3.0: 'Somewhat helpful',
                                       4.0: 'Not that helpful',
                                       5.0: 'Not helpful at all',
                                       6.0: 'Unsure',
                                       7.0: 'Prefer not to say'}).copy()

print(data['32_mentor_helpfulness'].value_counts())

ses_indicators = ['38_food_bank_use','38_social_assistance','38_work_to_support_family']
# create a new column low_ses, 0 if non checked off, 1 if at least one checked off

# --- Filter and Clean --
def bin_helpfulness(x):
    if x in ['Very helpful', 'Fairly helpful', 'Somewhat helpful']:
        return 1
    elif x in ['Not that helpful', 'Not helpful at all']:
        return 0
    else:
        return None # exclude
    

data['mentor_helpful_bin'] = data['32_mentor_helpfulness'].apply(bin_helpfulness)
data = data.dropna(subset=['mentor_helpful_bin'])

print(data['mentor_helpful_bin'].value_counts())



# --- see if those who are low socioeconmic find mentor helpful ---

# Work from the original data without modifying it
ses_indicators = ['38_food_bank_use','38_social_assistance','38_work_to_support_family']

# Filter first, on the original codes (1 = hardship, 2 = no hardship)
logistic_data = data[
    (
        (data['38_food_bank_use'] == 1) |
        (data['38_social_assistance'] == 1) |
        (data['38_work_to_support_family'] == 1)
    ) |
    (
        (data['38_food_bank_use'] == 2) &
        (data['38_social_assistance'] == 2) &
        (data['38_work_to_support_family'] == 2)
    )
].copy()

# Now recode SES
logistic_data[ses_indicators] = logistic_data[ses_indicators].replace({2: 0})
# Recreate low_ses indicator
logistic_data['low_ses'] = logistic_data[ses_indicators].any(axis=1).astype(int)

# --- Logistic Model ---
target = logistic_data['mentor_helpful_bin'].astype(int)


n_total = len(target)
n_pos = (target == 1).sum()  
n_neg = (target == 0).sum()  
w_pos = n_total / (2 * n_pos)  
w_neg = n_total / (2 * n_neg) 

X = logistic_data['low_ses']
X = sm.add_constant(X)  # Add intercept

logistic_data['asst_weights'] = target.apply(lambda x: w_pos if x == 1 else w_neg)

log_reg = sm.GLM(target, X, family=sm.families.Binomial(), freq_weights=logistic_data['asst_weights']).fit()

print("see if those who are low socioeconmic find mentor helpful",
      log_reg.summary())

# youth from low SES groups have a 78% lower odds of finding their mentor helpful, compared to higher SES groups

# now look at how being from low SES group effects acess

print(logistic_data['low_ses'].value_counts())
