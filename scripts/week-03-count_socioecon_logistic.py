# imports
import pandas as pd
import statsmodels.api as sm

# read in data 
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')

# columns to look at
q38_columns = ['38_social_assistance', '38_work_to_support_family', '38_food_bank_use']
q49_columns = ['49_adult_social_assistance']

# counts 
count1 = (data['38_social_assistance'] == 1).sum()
print('count of those whose guardians required social assitance at 12-18:', count1)

count2 = (data['49_adult_social_assistance'] == 1).sum()
print('count of those who required social assitance since turning 18:', count2)

count_both = ((data['38_social_assistance'] == 1) & (data['49_adult_social_assistance'] == 1)).sum()
print("count of those who required social assitance between 12-18 and since turning 18:", count_both)

# keep only rows with valid 1/2 (Yes/No) responses
logistic_data = data[
    data['49_adult_social_assistance'].isin([1, 2]) & 
    data['38_social_assistance'].isin([1, 2])
].copy()

# convert 'Yes'/'No' to binary
logistic_data['18_early_mentor'] = logistic_data['18_early_mentor'].replace({'Yes': 1, 'No': 0})
logistic_data['19_teen_mentor'] = logistic_data['19_teen_mentor'].replace({'Yes': 1, 'No': 0})

# create mentor_present: 1 if either mentor is present, else 0
logistic_data['mentor_present'] = (
    (logistic_data['18_early_mentor'] == 1) | 
    (logistic_data['19_teen_mentor'] == 1)
).astype(int)

# recode past assistance and target to binary
logistic_data['38_social_assistance'] = logistic_data['38_social_assistance'].replace({2: 0})
target = logistic_data['49_adult_social_assistance'].replace({2: 0})

# build model input
X = logistic_data[['38_social_assistance', 'mentor_present']]
X = sm.add_constant(X)  # Add intercept

# fit logistic model
log_reg = sm.Logit(target, X).fit()
print(log_reg.summary())

# Significant result:
# parents/guardians requiring social assiatance or disability suppourt  during the ages of 12-18 are almost 6 (e^{1.8025}) times as likely
# to require social assitance since turning 18.


# parents/guardians requiring social assiatance or disability suppourt during the ages of 12-18: 204

# count of those who did not complete highschool:
count_no_hs = (data['12_highschool_ged'] == "No").sum()
print("count of those who did not complete highschool", count_no_hs)
# parents/guardians requiring social assiatance or disability suppourt during the ages of 12-18 and did not complete highschool
count_hs_social = ((data['38_social_assistance'] == 1) & (data['12_highschool_ged'] == "No")).sum()
print("parents/guardians requiring social assiatance or disability suppourt during the ages of 12-18 and completing highschool", count_hs_social)

# Filter to keep only Yes/No for high school
logistic_data = logistic_data[logistic_data['12_highschool_ged'].isin(['Yes', 'No'])]
# Recode to binary
logistic_data['12_highschool_ged'] = logistic_data['12_highschool_ged'].replace({'Yes': 1, 'No': 0})


target2 = logistic_data['12_highschool_ged']
X2 = logistic_data[['38_social_assistance', 'mentor_present']]
X2 = sm.add_constant(X2)
log_reg2 = sm.Logit(target2, X2).fit()

print(log_reg2.summary())

# is mentorship significant with social assitance after turning 18??
X3 = logistic_data[['mentor_present']]
X3 = sm.add_constant(X3)  # Add intercept

log_reg3 = sm.Logit(target.loc[X3.index], X3).fit()
print(log_reg3.summary())

# check = logistic_data[['49_adult_social_assistance','38_social_assistance']]
# for col in check.columns:
#     print(f"Unique values in {col}: {check[col].unique()}")


