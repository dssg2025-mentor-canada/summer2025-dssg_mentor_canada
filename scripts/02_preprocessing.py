# date: 2025-05-26
# A) More basic questions   

# HIGH Produce a clean excel table with the results from the 2838 responses  
# HIGH Produce descriptive statistics profiles using existing template for respondent categories:  
# Women, young men   
# Low income (possible proxies: Q38 life events: either or both: social assistance or disability support OR used food banks)  
# Highest parent education (Q11) no-PSE  
# LOW Produce descriptive statistics for mentors #2 and 3 (not most meaningful)  
# LOW Analysis of informal mentors only category  

# Outcomes:  
# Q12. High school GED  
# Q13. Further education/training  -> counts at each level 
# Q13a. Higher education (controlling for age)  
# Q14. Employment status  
# Q41. Career planning  
# Q43. Social capital  
# Q44. Help-seeking  
# Q45. Mental health  
# Q46. Mental well-being  
# Q27. Belonging  
# Q50. Volunteering  

# Variables:
# Age = Q1
# Province = Q2
# First nation? = Q3
# Ethnicity = Q4
# Gender = Q6
# Disability at any point in life = Q9, Q9a (diagnosed) 
# Who's your primary caregiver = Q10
# Parent education = Q11
# Your education = Q12, 13
# Currently a student or working or both = Q14
# Prev mentor exp = Q18, Q19
# Barrier to mentor growing up = Q19 c1
# MENTORING relationship quality = Q27 (e.g., …a trusting/warm/close relationship?)
# Level of support given by mentor = Q29, 30 (more support can be quantified as more items being selected)
# Preceived importance/helpfulness of the support = Q30a, 32
# Negative experience during mentorship = Q33, 34
# Life quality during teen year = Q36
# The kinds of support received during teen = Q37a
# Negative life events during teen = Q38
# Kinds of social support you might need currently = Q43
# Current mental health rating (adult) = Q45
# Current recent well-being rating (adult) = Q46, 47 (belonging)
# Negative life events during adulthood = Q49
# Recent volunterring engagement / volunteered as mentor (adult) = Q50, 50b; Q51, 51a, 51b
# Becoming a mentor intent as an adult = Q52

import pandas as pd
import os
from siuba import _, group_by, summarize, filter, select, mutate, arrange, count

youth = pd.read_csv('../../dssg-2025-mentor-canada/Data/intermediate.csv')
youth_tidy = pd.DataFrame(index = youth.index)

### Previously reversed one-hot encoding (as a reference):
# gender_cols = youth.loc[:,'QS1_9_GENDER1_1_1':'QS1_9_GENDER1_6_6']
# youth['QS1_9_gender'] = gender_cols.idxmax(axis = 1)
# youth['QS1_9_gender'].head().reset_index()

# Ethnicity
ethnicity_cols = youth.loc[:, 'QS1_6_ETHNOCULTURAL1_1_1':'QS1_6_ETHNOCULTURAL1_14_14'].columns
youth[ethnicity_cols] = youth[ethnicity_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy['4_ethnicity'] = youth[ethnicity_cols].idxmax(1)

# Gender Identity
gender_cols = youth.loc[:,'QS1_9_GENDER1_1_1':'QS1_9_GENDER1_6_6'].columns
youth[gender_cols] = youth[gender_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy['6_gender_indentity'] = youth[gender_cols].idxmax(1)

# Preceived / Subclinical Disability

# Diagnosed Disability
