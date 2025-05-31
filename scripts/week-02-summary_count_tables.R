library(tidyverse)
library(janitor)

youth <- read_rds('../../dssg-2025-mentor-canada/Data/processed_youth_factorized.RDS')
glimpse(youth)

# Transform `0_postalcode_fsa` into all upper case letters.
youth <- youth |>
         mutate(`0_postalcode_fsa` = toupper(`0_postalcode_fsa`))

# 1. Summary count table for some variables of interest:
# Empty vector and tibble to put the result!
result_list_of_df <- list()
result_df <- data.frame()
colnames(youth)
cols_to_count <- c('2_province', '2b_community_type', '3_indigenous_status',
                   '4_ethnicity', '5_newcomer', '6_gender_identity',
                   '7_trans_ident', '8_sexual_orient', '9_subclinical_disability',
                   '9a_diagnosed_disability', '10_caregiver', '12_highschool_ged',
                   '13_further_edu', '13a_further_edu_level', '14_employment')

mentor_cols_to_count <- c('16_early_meaningful_person', '17_teen_meaningful_person', '18_early_mentor',
                          '18a_early_mentor_form', '18a1_early_mentor_exp', '18d_early_mentor_unmet_access',
                          '19_teen_mentor', '19c_teen_mentor_unmet_access', '20c_teen_mentor1_form',
                          '20c_teen_mentor2_form', '20c_teen_mentor3_form', '40_adult_mentor')

# Function 1 (count_loop) description: for every specified column, group_by() is and then apply summarize n() to it. 

# Function 1:
count_loop <- function(data, list_cols_to_count) {
  for (col in list_cols_to_count) {
    one_n_df <- data |>
      filter(!is.na(.data[[col]])) |> 
      group_by(.data[[col]]) |>
      summarize(count = n()) 
    
    result_list_of_df[[col]] <- one_n_df
  }
  return(result_list_of_df)
}

# Use count_loop function with columns specified in `cols_to_count`:
list_of_df_spec_cols <- count_loop(data = youth, list_cols_to_count = cols_to_count)
# Use count_loop function with columns specified in `mentor_cols_to_count`:
count_loop(data = youth, list_cols_to_count = mentor_cols_to_count)

# Function 2 (widen_count_df) description:
# Attach column name to each grouped by (group_by) categorical level in each count table
# within a list of count tables. Each count table is then applied with pivot_wider().
# All the wide tables is then combined into one large dataframe. 

combined_count_df <- data.frame()

for (df in list_of_df_spec_cols) { # draft for function (to be deleted later)
  col_name <- colnames(df)[1]
  count_col <- colnames(df)[2]
  wider_df <- df |>
    pivot_wider(names_from = all_of(col_name),
                values_from = all_of(count_col),
                names_prefix = col_name) |>
    janitor::clean_names()
  
  if (nrow(combined_count_df) == 0) {
    combined_count_df <- wider_df}
  else {
    combined_count_df <- cbind(combined_count_df, wider_df)}
  }


widen_count_df <- function(list_of_count_df) {
  for (df in list_of_count_df) {
    col_name <- colnames(df)[1]
    count_col <- colnames(df)[2]
    wider_df <- df |>
                pivot_wider(names_from = all_of(col_name),
                            values_from = all_of(count_col),
                            names_prefix = col_name) |>
                janitor::clean_names()
    
    if (nrow(combined_count_df) == 0) {
      combined_count_df <- wider_df
      }
    else {
      combined_count_df <- cbind(combined_count_df, wider_df)}
    }
  return(combined_count_df)
}

widen_count_df(list_of_df_spec_cols)

