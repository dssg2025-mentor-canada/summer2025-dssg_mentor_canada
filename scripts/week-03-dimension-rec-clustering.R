library(tidyverse)
library(Gifi)
library(psych)
library(tidyclust)

youth <- read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')
glimpse(youth)
ncol(youth)
# This script experiments with converting ordinal column.

# Define all columns in order --------
columns <- c("0_postalcode_fsa", "1_age", "2_province", "2b_community_type", "3_indigenous_status", 
             "4_ethnicity", "5_newcomer", "6_gender_identity", "7_trans_ident", "8_sexual_orient", 
             "9_subclinical_disability", "9a_diagnosed_disability", "10_caregiver", "11_birth_mother_edu", 
             "11_birth_father_edu", "12_highschool_ged", "13_further_edu", "13a_further_edu_level", 
             "14_employment", "15_yearly_income", "16_early_meaningful_person", "17_teen_meaningful_person", 
             "18_early_mentor", "18a_early_mentor_form", "18a1_early_mentor_exp", "18c_early_mentor_seek", 
             "18d_early_mentor_unmet_access", "19_teen_mentor", "19a_teen_mentor_n", "19b_teen_mentor_seek", 
             "19c_teen_mentor_unmet_access", "19c1_access_barriers", "20b_teen_mentor1_relation", 
             "20c_teen_mentor1_form", "20d_teen_mentor1_type", "20e_teen_mentor1_location", 
             "20f_teen_mentor1_duration", "20g_teen_mentor1_experience", "20h_teen_mentor1_focus", 
             "20i_teen_mentor1_geolocation", "20b_teen_mentor2_relation", "20c_teen_mentor2_form", 
             "20d_teen_mentor2_type", "20e_teen_mentor2_location", "20f_teen_mentor2_duration", 
             "20g_teen_mentor2_experience", "20h_teen_mentor2_focus", "20i_teen_mentor2_geolocation", 
             "20b_teen_mentor3_relation", "20c_teen_mentor3_form", "20d_teen_mentor3_type", 
             "20e_teen_mentor3_location", "20f_teen_mentor3_duration", "20g_teen_mentor3_experience", 
             "20h_teen_mentor3_focus", "20i_teen_mentor3_geolocation", "23_mentor1_init", "23_mentor2_init", 
             "23a_teen_mentor_init_reason", 
             "24a_teen_mentor_prefer", "26_language_similar", 
             "26_gender_ident_similar", "26_ethnic_similar", "26_religion_similar", "26_sex_ori_similar", 
             "27_relation_trusting", "27_relation_warm", "27_relation_close", "27_relation_happy", 
             "27_relation_respectful", "28_problem_solve", "28_listening", "28_company", "28_contact_freq", 
             "28_enjoyment", "28_understanding", "28_acceptance", "28_involvement", "28_trust", 
             "28_opinion", "28_fun", "28_planning", "28_teaching", "28_future", "28_reassurance", 
             "28_attention", "28_respect", "28_proactive", "28_patient", "28_familial", 
             "28_similar_interest", "28_similarity", "29_transition_stay_school", "29_transition_new_school", 
             "29_transition_new_community", "29_transition_license", "29_transition_job_aspiration", 
             "29_transition_first_job", "29_transition_higher_edu", "29_transition_independence", 
             "29_transition_funding_higher_edu", "29_transition_none", "29_transition_other", 
             "29_transition_prefer_not_say", "31_school_interest", "31_school_involvement", 
             "31_leadership", "31_social_skill", "31_self_pride", "31_confidence", "31_hope", 
             "31_self_knowledge", "31_direction", "32_mentor_helpfulness", "34_negative_experience", 
             "36a_teen_capability", "36b_teen_perceived_failure", "36c_teen_happiness", 
             "36d_teen_ident_alignment", "36e_teen_shame", "36f_teen_contentment", 
             "36g_teen_identity_goal", "36h_teen_lack_pride", "38_parent_prison", "38_school_absence", 
             "38_school_repeat", "38_school_suspended", "38_criminal_record", "38_freq_school_change", 
             "38_lack_school_access", "38_early_parenthood", "38_social_assistance", 
             "38_care_for_family", "38_work_to_support_family", "38_early_homelessness", 
             "38_food_bank_use", "38_youth_in_care", "40_adult_mentor", "40a_adult_mentor_experience", 
             "43_household_help_access", "43_financial_advice_access", "43_emotional_support_access", 
             "43_career_advice_access", "45_mental_health_rating", "46_optimism", 
             "46_perceived_capability", "46_ease_going", "46_problem_solve", "46_mental_clarity", 
             "46_relatedness_to_others", "46_decisiveness", "47_belonging", "49_adult_arrested", 
             "49_adult_prison", "49_adult_social_assistance", "49_adult_child_services", 
             "49_adult_homeless", "49_adult_food_banks", "51_adult_being_mentor", 
             "51a_adult_being_mentor_form")

# Remove 0_postalcode_fsa ------
youth <- youth |> 
  select(-`...1`,-`0_postalcode_fsa`, -`23a_teen_mentor_init_reason`, -`23_mentor2_init`)

# Define metric and ordinal variables ------
metric_vars <- c("1_age", "15_yearly_income", "19a_teen_mentor_n")
ordinal_vars <- c("11_birth_mother_edu", "11_birth_father_edu", "13a_further_edu_level", 
                  "18a1_early_mentor_exp", "18c_early_mentor_seek", "19b_teen_mentor_seek", 
                  "20d_teen_mentor1_type", "20e_teen_mentor1_location", "20f_teen_mentor1_duration", 
                  "20g_teen_mentor1_experience", "20f_teen_mentor2_duration", "20f_teen_mentor3_duration", 
                  "20g_teen_mentor2_experience", "20g_teen_mentor3_experience", "26_language_similar", 
                  "26_gender_ident_similar", "26_ethnic_similar", "26_religion_similar", 
                  "26_sex_ori_similar", "27_relation_trusting", "27_relation_warm", 
                  "27_relation_close", "27_relation_happy", "27_relation_respectful", 
                  "28_problem_solve", "28_listening", "28_company", "28_contact_freq", 
                  "28_enjoyment", "28_understanding", "28_acceptance", "28_involvement", 
                  "28_trust", "28_opinion", "28_fun", "28_planning", "28_teaching", 
                  "28_future", "28_reassurance", "28_attention", "28_respect", 
                  "28_proactive", "28_patient", "28_familial", "28_similar_interest", 
                  "28_similarity", "31_school_interest", "31_school_involvement", 
                  "31_leadership", "31_social_skill", "31_self_pride", "31_confidence", 
                  "31_hope", "31_self_knowledge", "31_direction", "32_mentor_helpfulness", 
                  "36a_teen_capability", "36b_teen_perceived_failure", "36c_teen_happiness", 
                  "36d_teen_ident_alignment", "36e_teen_shame", "36f_teen_contentment", 
                  "36g_teen_identity_goal", "36h_teen_lack_pride", "38_parent_prison", 
                  "38_school_absence", "38_school_repeat", "38_school_suspended", 
                  "38_criminal_record", "38_freq_school_change", "38_lack_school_access", 
                  "38_early_parenthood", "38_social_assistance", "38_care_for_family", 
                  "38_work_to_support_family", "38_early_homelessness", "38_food_bank_use", 
                  "38_youth_in_care", "40_adult_mentor", "40a_adult_mentor_experience", 
                  "43_household_help_access", "43_financial_advice_access", 
                  "43_emotional_support_access", "43_career_advice_access", 
                  "45_mental_health_rating", "46_optimism", "46_perceived_capability", 
                  "46_ease_going", "46_problem_solve", "46_mental_clarity", 
                  "46_relatedness_to_others", "46_decisiveness", "47_belonging", 
                  "49_adult_arrested", "49_adult_prison", "49_adult_social_assistance", 
                  "49_adult_child_services", "49_adult_homeless", "49_adult_food_banks")

#----------------------------------------------
# RUN THIS: draft for only ordinal MCA

youth_ordinal <- youth |>
  subset(select = ordinal_vars) |>
  select(`11_birth_mother_edu`, `11_birth_father_edu`, `18c_early_mentor_seek`)
# youth <- youth |>
#   mutate(across(all_of(ordinal_vars), 
#                 ~as.factor(as.ordered(.)))) 

# Ensure variables are ordered factors
youth_ordinal <- youth_ordinal |>
  mutate(across(everything(), ~as.factor(as.ordered(.))))

# Impute missing values for ordinal variables (mode)
youth_ordinal <- youth_ordinal |>
  mutate(across(where(is.factor), ~ifelse(is.na(.), names(which.max(table(., useNA = "no"))), .)))

# truncate
youth_ordinal <- youth_ordinal |>
slice(1:12)


glimpse(youth)
str(youth)
result <- homals(youth_ordinal)
summary(princals_result)
plot(princals_result)

# ---------------------------------------------

# categorical / nominal  variables:
nominal_vars <- setdiff(names(youth), c(ordinal_vars, metric_vars))
lapply(youth[nominal_vars], table)  # check

# make scale level vector for subsequent Gifi `levels = ` argument:
scale_levels <- ifelse(columns %in% metric_vars, "numeric")
                       
scale_levels <- ifelse(columns %in% metric_vars, "numeric",
                       ifelse(columns %in% ordinal_vars, "ordinal", "nominal"))


# Verify length
length(scale_levels)  

glimpse(youth)
str(youth)

# convert each ordinal column to numeric for Gifi:
youth <- youth |>
  mutate(across(all_of(ordinal_vars), 
                ~as.numeric(as.ordered(.)))) 

# Convert nominal columns to factors
youth <- youth |>
  mutate(across(all_of(nominal_vars), as.factor))
# Revert back to dummy
# youth <- fastDummies::dummy_cols(youth, select_columns = nominal_vars, remove_selected_columns = TRUE)

# # Ensure all columns are numeric
# youth <- youth |>
#   mutate(across(everything(), as.numeric))

# Impute missing values for metric variables (mean)
youth <- youth |>
  mutate(across(all_of(c(metric_vars,ordinal_vars)), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Impute missing values for nominal and ordinal variables (mode)
youth <- youth |>
  mutate(across(all_of(c(nominal_vars)), 
                ~ifelse(is.na(.), levels(.)[which.max(table(.))], .)))


# Create scale_levels for all columns
scale_levels <- vector("character", length = ncol(youth))
names(scale_levels) <- names(youth)
scale_levels[names(youth) %in% metric_vars] <- "numeric"
scale_levels[names(youth) %in% ordinal_vars] <- "ordinal"
scale_levels[names(youth) %in% nominal_vars] <- "nominal"


# Gifi ------- runs error:
# result <- princals(youth, levels = scale_levels)
# summary(princals_result)
# plot(princals_result)
