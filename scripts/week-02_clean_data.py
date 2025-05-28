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

# Load the data
youth = pd.read_csv('../../dssg-2025-mentor-canada/Data/encodedselectall.csv')

# Dictionary to collect all new columns
youth_tidy_cols = {}

# Ethnicity
ethnicity_cols = youth.loc[:, 'Race_SouthAsian':'Race_Unsure'].columns
youth[ethnicity_cols] = youth[ethnicity_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['4_ethnicity'] = youth[ethnicity_cols].idxmax(1)

# Gender Identity
gender_cols = youth.loc[:, 'Gender_Woman':'Gender_Other'].columns
youth[gender_cols] = youth[gender_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['6_gender_identity'] = youth[gender_cols].idxmax(1)

# Perceived / Subclinical Disability
youth_tidy_cols['9_subclinical_disability'] = youth['QS1_12_DISABIL']
# Diagnosed Disability
youth_tidy_cols['9a_diagnosed_disability'] = youth['QS1_13_DISABIL']

# Primary caregiver growing up
caregiver_cols = youth.loc[:, 'Primary_BirthMother':'Primary_Other'].columns
youth[caregiver_cols] = youth[caregiver_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['10_caregiver'] = youth[caregiver_cols].idxmax(1)

# Caregiver's educational level/attainment
youth_tidy_cols['11_birth_mother_edu'] = youth['QS1_18_PARENTEDUC1'].astype('Int64')
youth_tidy_cols['11_birth_father_edu'] = youth['QS1_18_PARENTEDUC2'].astype('Int64')

# Kinds of Education attainment
youth_tidy_cols['12_highschool_ged'] = youth['QS1_19_HIGHSCHOOL']
youth_tidy_cols['13_further_edu'] = youth['QS1_21_FURTHEDUCA']
youth_tidy_cols['13a_further_edu_level'] = youth['QS1_22_HIGHESTEDU']

# Current employment status
youth_tidy_cols['14_employment'] = youth['QS1_25_EMPLOYMENT']

# Yearly income estimate
youth['15_yearly_income'] = youth['QS1_28_EMPLOYMENT_calculated'].fillna(0)
# youth['15b_yearly_from_month'] = youth['Month_income'].fillna(0) * 12
# youth['yearly_from_seimimonth'] = youth['Semimonth_income'].fillna(0) * 24
# youth['yearly_from_biweek'] = youth['Biweek_income'].fillna(0) * 26
# youth['yearly_from_week'] = youth['Week_income'].fillna(0) * 52
# youth['total_yearly_income'] = (
#     youth['yearly_from_month'] +
#     youth['yearly_from_seimimonth'] +
#     youth['yearly_from_biweek'] +
#     youth['yearly_from_week']
# )
youth_tidy_cols['15_yearly_income'] = youth['15_yearly_income']

# Presence of meaningful person in early life 6-11
youth_tidy_cols['early_meaningful_person'] = youth['QS2_1_MEANINGFULP']

# Presence of meaningful person in teen years 12-18
youth_tidy_cols['teen_meaningful_person'] = youth['QS2_2_MEANINGFULP']

# Presence of mentor in early life 6-11
youth_tidy_cols['early_mentor'] = youth['QS2_3_PRESENCEOFM']
# Formal/informal mentor format in early life 6-11
youth_tidy_cols['early_mentor_form'] = youth['QS2_4_MENTOR61FOR']
# Rating mentor experience in early life 6-11
youth_tidy_cols['early_mentor_exp'] = youth['QS2_6_MENTOREXPER']
# Mentor seeking in early life 6-11
youth_tidy_cols['early_mentor_seek'] = youth['QS2_7_MENTOR611SE']
# Difficult mentor access in early life 6-11
youth_tidy_cols['early_mentor_unmet_access'] = youth['QS2_8_UNMETNEED61']

# Presence of mentor in teen years 12-18
youth_tidy_cols['teen_mentor'] = youth['QS2_9_PRESENCEOFA']
# Number of mentors in teen years 12-18
youth_tidy_cols['teen_mentor_n'] = youth['QS2_10_NUMBEROFME']
# Mentor seeking in teen years 12-18
youth_tidy_cols['teen_mentor_seek'] = youth['QS2_11_MENTOR1218']
# Unmet needs in teen years 12-18
youth_tidy_cols['teen_mentor_unmet_access'] = youth['QS2_12_UNMETNEED1']
# Access barrier to mentor in teen years 12-18
teen_access_barrier_cols = youth.loc[:, 'Barrier_Parent':'Barrier_PreferNotToSay'].columns
youth[teen_access_barrier_cols] = youth[teen_access_barrier_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['19c_access_barriers'] = youth[teen_access_barrier_cols].idxmax(1)

# Mentor 1
mentor_figure_cols = youth.loc[:, 'Relation1_SchoolStaff':'Relation1_Other'].columns
youth[mentor_figure_cols] = youth[mentor_figure_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20b_teen_mentor1_figure'] = youth[mentor_figure_cols].idxmax(1)
youth_tidy_cols['20c_teen_mentor1_form'] = youth['QS2_16_FORMAT_1']
youth_tidy_cols['20d_teen_mentor1_type'] = youth['QS2_17_TYPE_1']
youth_tidy_cols['20e_teen_mentor1_location'] = youth['QS2_18_LOCATION_1']
youth_tidy_cols['20f_teen_mentor1_duration'] = youth['QS2_19_DURATION_1']
youth_tidy_cols['20g_teen_mentor1_experience'] = youth['QS2_20_EXPERIENCE_1']
teen_service_focus_cols = youth.loc[:, 'Focus1_EducationSupport':'Focus1_Emotional-SocialSupport'].columns
youth[teen_service_focus_cols] = youth[teen_service_focus_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20h_teen_mentor1_focus'] = youth[teen_service_focus_cols].idxmax(1)
youth_tidy_cols['20g_teen_mentor1_canada'] = youth['QS2_22_GEOLOCATI1']

# Mentor 2
mentor_figure_cols = youth.loc[:, 'Relation2_SchoolStaff':'Relation2_Other'].columns
youth[mentor_figure_cols] = youth[mentor_figure_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20b_teen_mentor2_figure'] = youth[mentor_figure_cols].idxmax(1)
youth_tidy_cols['20c_teen_mentor2_form'] = youth['QS2_16_FORMAT_2']
youth_tidy_cols['20d_teen_mentor2_type'] = youth['QS2_17_TYPE_2']
youth_tidy_cols['20e_teen_mentor2_location'] = youth['QS2_18_LOCATION_2']
youth_tidy_cols['20f_teen_mentor2_duration'] = youth['QS2_19_DURATION_2']
youth_tidy_cols['20g_teen_mentor2_experience'] = youth['QS2_20_EXPERIENCE_2']
teen_service_focus_cols = youth.loc[:, 'Focus2_EducationSupport':'Focus2_Emotional-SocialSupport'].columns
youth[teen_service_focus_cols] = youth[teen_service_focus_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20h_teen_mentor2_focus'] = youth[teen_service_focus_cols].idxmax(1)
youth_tidy_cols['20g_teen_mentor2_canada'] = youth['QS2_22_GEOLOCATI2']

# Mentor 3
mentor_figure_cols = youth.loc[:, 'Relation3_SchoolStaff':'Relation3_Other'].columns
youth[mentor_figure_cols] = youth[mentor_figure_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20b_teen_mentor3_figure'] = youth[mentor_figure_cols].idxmax(1)
youth_tidy_cols['20c_teen_mentor3_form'] = youth['QS2_16_FORMAT_3']
youth_tidy_cols['20d_teen_mentor3_type'] = youth['QS2_17_TYPE_3']
youth_tidy_cols['20e_teen_mentor3_location'] = youth['QS2_18_LOCATION_3']
youth_tidy_cols['20f_teen_mentor3_duration'] = youth['QS2_19_DURATION_3']
youth_tidy_cols['20g_teen_mentor3_experience'] = youth['QS2_20_EXPERIENCE_3']
teen_service_focus_cols = youth.loc[:, 'Focus3_EducationSupport':'Focus3_Emotional-SocialSupport'].columns
youth[teen_service_focus_cols] = youth[teen_service_focus_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['20h_teen_mentor3_focus'] = youth[teen_service_focus_cols].idxmax(1)
youth_tidy_cols['20g_teen_mentor3_canada'] = youth['QS2_22_GEOLOCATI3']

# Who initiated the mentor program
youth_tidy_cols['23_mentor1_init'] = youth['QS2_25_YOUTHINIT1']
youth_tidy_cols['23_mentor2_init'] = youth['QS2_25_YOUTHINIT2']
# Reason for mentorship initiation
init_reason_cols = youth.loc[:, 'Initated_StrugglingInSchool':'Initated_Other'].columns
youth[init_reason_cols] = youth[init_reason_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['23a_teen_mentor_init_reason'] = youth[init_reason_cols].idxmax(1)
# Teen's preferences about choosing mentor
mentor_prefer_cols = youth.loc[:, 'Match_GenderIdentity':'Match_PreferNotToSay'].columns
youth[mentor_prefer_cols] = youth[mentor_prefer_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['24a_teen_mentor_prefer'] = youth[mentor_prefer_cols].idxmax(1)
# Rate how well the teen's mentor matching was
youth_tidy_cols['26_language_similar'] = youth['QS2_30_MATCHSIMILAR1_1_1']
youth_tidy_cols['26_gender_ident_similar'] = youth['QS2_30_MATCHSIMILAR1_2_2']
youth_tidy_cols['26_ethnic_similar'] = youth['QS2_30_MATCHSIMILAR1_3_3']
youth_tidy_cols['26_religion_similar'] = youth['QS2_30_MATCHSIMILAR1_4_4']
youth_tidy_cols['26_sex_ori_similar'] = youth['QS2_30_MATCHSIMILAR1_5_5']
# Mentorship quality perceived
youth_tidy_cols['27_relation_trusting'] = youth['QS2_30_MATCHSIMILAR1_1_1']
youth_tidy_cols['27_relation_warm'] = youth['QS2_30_MATCHSIMILAR1_2_2']
youth_tidy_cols['27_relation_close'] = youth['QS2_30_MATCHSIMILAR1_3_3']
youth_tidy_cols['27_relation_happy'] = youth['QS2_30_MATCHSIMILAR1_4_4']
youth_tidy_cols['27_relation_respectful'] = youth['QS2_30_MATCHSIMILAR1_5_5']
# Mentorship engagement (agree/disagree)
youth_tidy_cols['28_problem_solve'] = youth['QS2_30_MATCHSIMILAR1_1_1']
youth_tidy_cols['28_listening'] = youth['QS2_32_MENTORINGENG1_2_2']
youth_tidy_cols['28_company'] = youth['QS2_32_MENTORINGENG1_3_3']
youth_tidy_cols['28_contact_freq'] = youth['QS2_32_MENTORINGENG1_4_4']
youth_tidy_cols['28_enjoyment'] = youth['QS2_32_MENTORINGENG1_5_5']
youth_tidy_cols['28_understanding'] = youth['QS2_32_MENTORINGENG1_6_6']
youth_tidy_cols['28_acceptance'] = youth['QS2_32_MENTORINGENG1_7_7']
youth_tidy_cols['28_involvement'] = youth['QS2_32_MENTORINGENG1_8_8']
youth_tidy_cols['28_trust'] = youth['QS2_32_MENTORINGENG1_9_9']
youth_tidy_cols['28_opinion'] = youth['QS2_32_MENTORINGENG1_10_10']
youth_tidy_cols['28_fun'] = youth['QS2_32_MENTORINGENG1_11_11']
youth_tidy_cols['28_planning'] = youth['QS2_32_MENTORINGENG1_12_12']
youth_tidy_cols['28_teaching'] = youth['QS2_32_MENTORINGENG1_13_13']
youth_tidy_cols['28_future'] = youth['QS2_32_MENTORINGENG1_14_14']
youth_tidy_cols['28_reassurance'] = youth['QS2_32_MENTORINGENG1_15_15']
youth_tidy_cols['28_attention'] = youth['QS2_32_MENTORINGENG1_16_16']
youth_tidy_cols['28_respect'] = youth['QS2_32_MENTORINGENG1_17_17']
youth_tidy_cols['28_proactive'] = youth['QS2_32_MENTORINGENG1_18_18']
youth_tidy_cols['28_patient'] = youth['QS2_32_MENTORINGENG1_19_19']
youth_tidy_cols['28_familial'] = youth['QS2_32_MENTORINGENG1_20_20']
youth_tidy_cols['28_similar_interest'] = youth['QS2_32_MENTORINGENG1_21_21']
youth_tidy_cols['28_similarity'] = youth['QS2_32_MENTORINGENG1_22_22']

# Transitions
youth_tidy_cols['29_transition_stay_school'] = youth['Transition_School']
youth_tidy_cols['29_transition_new_school'] = youth['Transition_NewSchool']
youth_tidy_cols['29_transition_new_community'] = youth['Transition_NewCommunity']
youth_tidy_cols['29_transition_license'] = youth['Transition_GettingDriversLicense']
youth_tidy_cols['29_transition_job_aspiration'] = youth['Transition_JobAspirations']
youth_tidy_cols['29_transition_first_job'] = youth['Transition_GettingFirstJob']
youth_tidy_cols['29_transition_higher_edu'] = youth['Transition_ApplyingToTradeSchool-Collge-Uni']
youth_tidy_cols['29_transition_independence'] = youth['Transition_IndependenceFromGuardian']
youth_tidy_cols['29_transition_funding_higher_edu'] = youth['Transition_FundingForTradeSchool-Collge-Uni']
youth_tidy_cols['29_transition_none'] = youth['Transition_NoneOfAbove']
youth_tidy_cols['29_transition_other'] = youth['Transition_Other']
youth_tidy_cols['29_transition_prefer_not_say'] = youth['Transition_PreferNotToSay']

# Skills taught by / influence from mentors
youth_tidy_cols['31_school_interest'] = youth['QS2_36_INFLUENCE1_1_1']
youth_tidy_cols['31_school_involvement'] = youth['QS2_36_INFLUENCE1_2_2']
youth_tidy_cols['31_leadership'] = youth['QS2_36_INFLUENCE1_3_3']
youth_tidy_cols['31_social_skill'] = youth['QS2_36_INFLUENCE1_4_4']
youth_tidy_cols['31_self_pride'] = youth['QS2_36_INFLUENCE1_5_5']
youth_tidy_cols['31_confidence'] = youth['QS2_36_INFLUENCE1_6_6']
youth_tidy_cols['31_hope'] = youth['QS2_36_INFLUENCE1_7_7']
youth_tidy_cols['31_self_knowledge'] = youth['QS2_36_INFLUENCE1_8_8']
youth_tidy_cols['31_direction'] = youth['QS2_36_INFLUENCE1_9_9']

# Mentor helpfulness
youth_tidy_cols['32_mentor_helpfulness'] = youth['QS2_37_HELPFULNESS']

# Negative mentor experience
mentor_neg_exp_cols = youth.loc[:, 'Negative_MetorQuit':'Negative_PreferNotToSay'].columns
youth[mentor_neg_exp_cols] = youth[mentor_neg_exp_cols].apply(pd.to_numeric, errors='coerce')
youth_tidy_cols['34_negative_experience'] = youth[mentor_neg_exp_cols].idxmax(1)

# Retrospective self-worth at age 12-18
unnamed_self_worth_cols = [
    'QS3_1_GLOBALSELFWOR1_1_1', 'QS3_1_GLOBALSELFWOR1_2_2', 'QS3_1_GLOBALSELFWOR1_3_3',
    'QS3_1_GLOBALSELFWOR1_4_4', 'QS3_1_GLOBALSELFWOR1_5_5', 'QS3_1_GLOBALSELFWOR1_6_6',
    'QS3_1_GLOBALSELFWOR1_7_7', 'QS3_1_GLOBALSELFWOR1_8_8'
]
named_self_worth_cols = [
    'teen_capability', 'teen_perceived_failure', 'teen_happiness', 'teen_ident_alignment',
    'teen_shame', 'teen_contentment', 'teen_identity_goal', 'teen_lack_pride'
]
for named_col, unnamed_col in zip(named_self_worth_cols, unnamed_self_worth_cols):
    youth_tidy_cols[named_col] = youth[unnamed_col]

# Negative life events
unnamed_neg_life_event_cols = [
    'QS3_4_LIFEEVENTS1_2_2', 'QS3_4_LIFEEVENTS1_3_3', 'QS3_4_LIFEEVENTS1_5_5',
    'QS3_4_LIFEEVENTS1_6_6', 'QS3_4_LIFEEVENTS1_7_7', 'QS3_4_LIFEEVENTS1_8_8',
    'QS3_4_LIFEEVENTS1_9_9', 'QS3_4_LIFEEVENTS1_11_11', 'QS3_4_LIFEEVENTS1_12_12',
    'QS3_4_LIFEEVENTS1_15_15', 'QS3_4_LIFEEVENTS1_16_16', 'QS3_4_LIFEEVENTS1_17_17',
    'QS3_4_LIFEEVENTS1_18_18'
]
named_neg_life_event_cols = [
    'parent_prison', 'school_absence', 'school_repeat', 'school_suspended', 'criminal_record',
    'freq_school_change', 'lack_school_access', 'early_parenthood', 'caregiver_role',
    'work_to_support_family', 'early_homelessness', 'food_bank_use', 'youth_in_care'
]
for named_col, unnamed_col in zip(named_neg_life_event_cols, unnamed_neg_life_event_cols):
    youth_tidy_cols[named_col] = youth[unnamed_col]

# Adult
youth_tidy_cols['adult_mentor'] = youth['QS4_1_MEANINGFULPERSON']
youth_tidy_cols['adult_mentor_experience'] = youth['QS4_2_MEANINGFULPERSON']

# Current social capital / access / quality of life
unnamed_social_cap_cols = [
    'QS4_7_SOCIALCAPITAL1_1_1', 'QS4_7_SOCIALCAPITAL1_2_2',
    'QS4_7_SOCIALCAPITAL1_3_3', 'QS4_7_SOCIALCAPITAL1_4_4'
]
named_social_cap_cols = [
    'household_help_access', 'financial_advice_access',
    'emotional_support_access', 'career_advice_access'
]
for named_col, unnamed_col in zip(named_social_cap_cols, unnamed_social_cap_cols):
    youth_tidy_cols[named_col] = youth[unnamed_col]

# Current perceived mental health rating
youth_tidy_cols['mental_health_rating'] = youth['QS4_9_MENTALHEALTH']

# Current well-being
unnamed_well_being = [
    'QS4_10_MENTALWELLBE1_1_1', 'QS4_10_MENTALWELLBE1_2_2', 'QS4_10_MENTALWELLBE1_3_3',
    'QS4_10_MENTALWELLBE1_4_4', 'QS4_10_MENTALWELLBE1_5_5', 'QS4_10_MENTALWELLBE1_6_6',
    'QS4_10_MENTALWELLBE1_7_7'
]
named_well_being = [
    'optimism', 'perceived_capability', 'ease_going',
    'problem_solve', 'mental_clarity', 'relatedness_to_others', 'decisiveness'
]
for named_col, unnamed_col in zip(named_well_being, unnamed_well_being):
    youth_tidy_cols[named_col] = youth[unnamed_col]

# Current sense of belonging
youth_tidy_cols['belongingness'] = youth['QS4_11_BELONGING']

# Current negative life events
unnamed_adult_neg_events = [
    'QS4_13_LIFEEVE1_1_1', 'QS4_13_LIFEEVE1_2_2', 'QS4_13_LIFEEVE1_3_3',
    'QS4_13_LIFEEVE1_4_4', 'QS4_13_LIFEEVE1_5_5', 'QS4_13_LIFEEVE1_6_6'
]
named_adult_neg_events = [
    'adult_arrested', 'adult_prison', 'adult_social_assistance',
    'adult_child_services', 'adult_homeless', 'adult_food_banks'
]
for named_col, unnamed_col in zip(named_adult_neg_events, unnamed_adult_neg_events):
    youth_tidy_cols[named_col] = youth[unnamed_col]

# Currently being a mentor
youth_tidy_cols['adult_being_mentor'] = youth['QS4_17_SERVEDASM']
# Currently being a formal/informal mentor
youth_tidy_cols['adult_being_mentor_form'] = youth['QS4_18_CURRENTOR']

# Create the tidy DataFrame using pd.concat
youth_tidy = pd.concat(youth_tidy_cols, axis=1)

# Ensure the index matches the original youth DataFrame
youth_tidy.index = youth.index