# --- Imports ---
import pandas as pd
import statsmodels.api as sm

# --- Load Data ---
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')

# --- Basic Counts ---
print('count of those whose guardians required social assistance (ages 12–18):', (data['38_social_assistance'] == 1).sum())
print('count of those who required social assistance since turning 18:', (data['49_adult_social_assistance'] == 1).sum())
print("count of those who required social assistance between 12–18 and since turning 18:",
      ((data['38_social_assistance'] == 1) & (data['49_adult_social_assistance'] == 1)).sum())
print("\nRaw target class distributions:")

print(data['12_highschool_ged'].value_counts())
print(data['49_adult_social_assistance'].value_counts())

# --- Filter and Clean ---

# remove observations with unsure/prefer not to say responses
logistic_data = data[
    data['49_adult_social_assistance'].isin([1, 2]) &
    data['38_social_assistance'].isin([1, 2]) &
    data['12_highschool_ged'].isin(['Yes', 'No'])
].copy()  

# binary encoding
logistic_data['12_highschool_ged'] = logistic_data['12_highschool_ged'].replace({'Yes': 1, 'No': 0})
logistic_data['38_social_assistance'] = logistic_data['38_social_assistance'].replace({2: 0})
logistic_data['49_adult_social_assistance'] = logistic_data['49_adult_social_assistance'].replace({2: 0})

# create mentor_present column
logistic_data['18_early_mentor'] = logistic_data['18_early_mentor'].replace({'Yes': 1, 'No': 0})
logistic_data['19_teen_mentor'] = logistic_data['19_teen_mentor'].replace({'Yes': 1, 'No': 0})

logistic_data['mentor_present'] = (
    (logistic_data['18_early_mentor'] == 1) | 
    (logistic_data['19_teen_mentor'] == 1)
).astype(int) # turns true to 1, false to 0


print('High School Completion:',logistic_data['12_highschool_ged'].value_counts())
print('Adult Social Assistance Use',logistic_data['49_adult_social_assistance'].value_counts())

# --- First Model ---

# create weights
target = logistic_data['49_adult_social_assistance']

n_total = len(target)
n_pos = (target == 1).sum()  
n_neg = (target == 0).sum()  
w_pos = n_total / (2 * n_pos)  
w_neg = n_total / (2 * n_neg) 

X = logistic_data[['38_social_assistance', 'mentor_present']]
X = sm.add_constant(X)  # Add intercept

logistic_data['asst_weights'] = target.apply(lambda x: w_pos if x == 1 else w_neg)

log_reg1 = sm.GLM(target, X, family=sm.families.Binomial(), freq_weights=logistic_data['asst_weights']).fit()

print("Predicting Adult Social Assistance Use Based on Parental Assistance and Mentorship Presence",
      log_reg1.summary())



# --- Second Model ---

target = logistic_data['12_highschool_ged']


n_total = len(target)
n_pos = (target == 1).sum()
n_neg = (target == 0).sum()

w_pos = n_total / (2 * n_pos)
w_neg = n_total / (2 * n_neg)

logistic_data['hs_weights'] = target.apply(lambda x: w_pos if x == 1 else w_neg)

X = sm.add_constant(logistic_data[['38_social_assistance', 'mentor_present']])

log_reg2 = sm.GLM(target, X, family=sm.families.Binomial(), freq_weights=logistic_data['hs_weights']).fit()

print("Predicting High School Completion Based on Mentorship Presence and Parental Social Assistance Use", log_reg2.summary())

# --- Third Model ---

target = logistic_data['49_adult_social_assistance']

n_total = len(target)
n_pos = (target == 1).sum()
n_neg = (target == 0).sum()

w_pos = n_total / (2 * n_pos)
w_neg = n_total / (2 * n_neg)

logistic_data['asst_weights'] = target.apply(lambda x: w_pos if x == 1 else w_neg)

X = sm.add_constant(logistic_data[['mentor_present']])

log_reg3 = sm.GLM(target, X, family=sm.families.Binomial(), freq_weights=logistic_data['asst_weights']).fit()

print("Assessing the Impact of Mentor Presence on Adult Social Assistance Use (Univariate Model)",log_reg3.summary())
