library(tidyverse)

youth <- read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv', col_select = -1)
head(youth)

# 11_birth_mother_edu, 11_birth_father_edu, 15_yearly_income
# 19a_teen_mentor_n
# Likert: 
# 26_language_similar	26_gender_ident_similar	26_ethnic_similar	26_religion_similar	
# 26_sex_ori_similar	27_relation_trusting	27_relation_warm	27_relation_close	27_relation_happy	
# 27_relation_respectful	28_problem_solve	28_listening	28_company	28_contact_freq	28_enjoyment	
# 28_understanding	28_acceptance	28_involvement	28_trust	28_opinion	28_fun	28_planning	
# 28_teaching	28_future	28_reassurance	28_attention	28_respect	28_proactive	28_patient	
# 28_familial	28_similar_interest	28_similarity

# 46_optimism	46_perceived_capability	46_ease_going	46_problem_solve	
# 46_mental_clarity	46_relatedness_to_others	46_decisiveness

# 49_adult_arrested	49_adult_prison	49_adult_social_assistance	
# 49_adult_child_services	49_adult_homeless	49_adult_food_banks

# * Please note that some variable's empty values have 
#   been imputed with 'No_Experience', which has not been validated.
youth <- youth |>
          mutate(across(`2_province`:`10_caregiver`, as_factor)) |>
          mutate(across(`12_highschool_ged`:`14_employment`, as_factor)) |>
          mutate(across(`16_early_meaningful_person`:`19_teen_mentor`, as_factor)) |>
          mutate(across(`19b_teen_mentor_seek`:`24a_teen_mentor_prefer`, as_factor)) |> 
          mutate(across(`29_transition_stay_school`:`31_direction`, as_factor)) |>
          mutate(across(`32_mentor_helpfulness`:`45_mental_health_rating`, as_factor)) |>
          mutate(`47_belonging` = as_factor(`47_belonging`)) |> 
          mutate(across(`51_adult_being_mentor`:`51a_adult_being_mentor_form`, as_factor))

glimpse(youth)

write_rds(youth, '../../dssg-2025-mentor-canada/Data/processed_youth_factorized.RDS')
  
  
  
  

