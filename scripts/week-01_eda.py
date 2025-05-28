
# %%
import pandas as pd
import matplotlib.pyplot as plt
from siuba import _, group_by, summarize, filter, select, mutate, arrange, count
df = pd.read_csv('../../dssg-2025-mentor-canada/Data/Data_2020-Youth-Survey.csv')

# %%
print(df.head())
print(df.info())
print(df.shape)

null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
print(null_counts)

# %%
df.columns
# drop columns up to AO (column 41)
df_dropped = df.iloc[:, 41:]
df_dropped.columns

# drop logic and validation columns
df_dropped = df_dropped.drop(columns=['QAge_Validation', 'Logic_QS1_6_Qtext', 'Logic_Qtext', 'QS1_8_Validation', 'Logic_QS1_26_Ask', 'QS1_29_Validation', 
                              'QS1_30_MValidatio', 'QS1_30_SMValidati', 'QS1_31_BWValidati', 'QS1_32_WValidatio', 'QS2_10_Validation', 'Logic_QS2_14_Ask','Logic_MENTORID1_1_1',
                              'Logic_MENTORID1_2_2', 'Logic_MENTORID1_3_3', 'Logic_AP_QS2_23', 'Logic_QS2_27_Ask', 'Logic_QS2_34_Valid', 'Logic_QS2_35_Ask', 'Logic_QS2_35_Mask1_1_1',
                              'Logic_QS2_35_Mask1_2_2', 'Logic_QS2_35_Mask1_3_3', 'Logic_QS2_35_Mask1_4_4', 'Logic_QS2_35_Mask1_5_5', 'Logic_QS2_35_Mask1_6_6', 'Logic_QS2_35_Mask1_7_7',
                              'Logic_QS2_35_Mask1_8_8', 'Logic_QS2_35_Mask1_9_9', 'Logic_QS2_35_Mask1_10_10', 'QS4_14_Validatio', 'QS4_15_Validatio', 'QS4_19_Validatio', 'QS4_23_Validatio'
                              ])

# %%
df_dropped.columns
# %%
text_columns = ['QS1_6_Other', 'QS1_9_Other', 'QS1_11_Other', 'QS1_16_Other', 'QS1_18_Other_1', 'QS1_18_Other_2', 'QS1_18_Other_3', 'QS1_18_Other_4', 'QS1_18_Other_5', 'QS1_18_Other_6', 'QS1_18_Other_7', 'QS1_18_Other_8', 'QS1_18_Other_9', 'QS1_18_Other_10', 
                'QS1_18_Other_11', 'QS1_22_Other', 'QS1_26_Other', 'QS1_27_Other', 'QS2_13_Other', 'QS2_14_MENTORID', 'QS2_14_MENTORID_2', 'QS2_14_MENTORID_3', 'QS2_18_LOCATION_1_O', 'QS2_15_RELATIONSHIP2', 'QS2_17_TYPE_2_Other', 'QS2_18_LOCATION_2_O', 'QS2_15_RELATIONSHIP3',
                'QS2_17_TYPE_3_Other', 'QS2_18_LOCATION_3_O', 'QS2_25_YOUTHINIT2', 'QS2_27_MENTORPROGRA2', 'QS2_33_TRANSITIONS_Ot', 'QS2_34_SUPPORTS_Ot', 'QS2_38_NETGATIVEMENTO', 'QS3_2_TRANSITIONWITHOUTMEN', 'QS3_3_TRANSITIONSWITHOUTMENTO', 'QS4_4_Other', 'QS4_5_SATEDU_Other']

# %%
# histogram of age
descriptive_columns = ['QS1_1_AGE', 'QS1_2_PROV', 'QS1_25_EMPLOYMENT', 'QS2_3_PRESENCEOFM', 'QS2_9_PRESENCEOFA', 'QS1_4_INDIGENOUS']
df_dropped['QS1_1_AGE'].plot.hist(bins=10, edgecolor='black')

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')

plt.show()

# %%
# table of number of observations for each province/territory
province_counts = df_dropped['QS1_2_PROV'].value_counts()
print(province_counts)
# %%
employment_counts = df_dropped['QS1_25_EMPLOYMENT'].value_counts()
print(employment_counts)

indigenous_counts = df_dropped['QS1_4_INDIGENOUS'].value_counts()
print(indigenous_counts)

# %%
df['QS1_25_EMPLOYMENT_abrivated'] = df['QS1_25_EMPLOYMENT'].replace({
    'Working (paid work for at least 1 hr/week)': 'Working',
    'Studying or in education/training': 'Studying',
    'Neither of the above': 'Neither',
    'Both': 'Both'
})

df['QS1_4_INDIGENOUS_abrivated'] = df['QS1_4_INDIGENOUS'].replace({
    "I don't identify as a member of these communities": 'Non-Indigenous',
    'First Nations (North American Indian)': 'First Nations',
    'Prefer not to say': 'Prefer not to say',
    'Unsure': 'Unsure',
    'Métis': 'Métis',
    'Inuk (Inuit)': 'Inuk'
})

pd.crosstab(df['QS1_4_INDIGENOUS_abrivated'], df['QS1_25_EMPLOYMENT_abrivated'])

# %%
presence_of_ment_611 = df_dropped['QS2_3_PRESENCEOFM'].value_counts()
presence_of_ment_1218 = df_dropped['QS2_9_PRESENCEOFA'].value_counts()

print("Presence of Mentor (ages 6-11) (QS2_3_PRESENCEOFM):")
print(presence_of_ment_611)

print("Presence of Mentor (ages 12-18) (QS2_3_PRESENCEOFM):")
print(presence_of_ment_1218)




# %%
