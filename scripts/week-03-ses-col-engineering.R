library(tidyverse)
library(GGally)
# `38_social_assistance`, `38_work_to_support_family`, `38_food_bank_use`

youth <- read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')
glimpse(youth)

# ---- Function to convert a range of columns into factors:----
factorized_col_range <- function(data, start_col, end_col) {
  mutated_df <- data |>
    mutate(across(.data[[start_col]]:.data[[end_col]], as_factor))
  return(mutated_df)
}
# --------------------------------------------------------------
# Teen:

# Convert all columns under Q38 into factor data type
youth <- youth |>
  factorized_col_range("38_parent_prison", "38_youth_in_care")

# Convert all columns under Q49 into factor data type
youth <- youth |>
  factorized_col_range("49_adult_arrested", "49_adult_food_banks")

# Counts:
## Count number of respondents to question 38 on `38_social_assistance`, `38_work_to_support_family`, `38_food_bank_use`
youth <- youth |>
  filter(`38_social_assistance` == 1 | `38_social_assistance` == 2) |>
  filter(`38_work_to_support_family` == 1 | `38_work_to_support_family` == 2) |> 
  filter(`38_food_bank_use` == 1 | `38_food_bank_use` ==  2) 
ses_early_count <- youth |>
  group_by(`38_social_assistance`, 
           `38_work_to_support_family`,
           `38_food_bank_use`) |>
  summarize(count = n())


total_respondant_to_Q38 <- ses_early_count |>
  ungroup() |>
  summarize(sum = sum(count))

ses_early_count <- ses_early_count |>
  ungroup() |>
  mutate(percentage_proportion = count / pull(total_respondant_to_Q38, 1) * 100)
ses_early_count
# combine selections into one single column:
ses_early_count <- ses_early_count |>
  mutate(selection = paste(`38_social_assistance`, 
                             `38_work_to_support_family`, 
                             `38_food_bank_use`)) |>
  mutate(selection = as_factor(selection))
write_csv(ses_early_count, "outputs/tables/week-03-count-youth-ses-indicator-responses.csv")

glimpse(ses_early_count)

# distribution
ses_early_count |>
  mutate(id = row_number()) |>
  ggplot(aes(y = selection, x = count, fill = selection)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 0, hjust = .5)) +
  labs(title = "Distribution of Responses to Question 38 as \n SES indicators as teen (age 12-18)",
       fill = "Response Selection",
       caption = "Response Selection Encoding for 3 questions: \n
       (a) Guardian(s) received social assistance support. \n
       (b) Respondent had to work a job to support family. \n
       (c) Experience of using the food banks. \n
       1 = Yes, 2 = No",
       y = "Possible Combination of Response Selections")  + 
  theme(plot.caption.position = "plot",
        plot.caption = element_text(hjust = 0))
ggsave("outputs/figures/week-03-ses-indicator-teen-response-count.png")


# ----------- Adult (current SES situation of the respondents):-------------
## Count number of respondents to question 49 on 
# `49_adult_social_assistance`, `49_adult_child_services`, `49_adult_food_banks`
youth <- youth |>
  filter(`49_adult_social_assistance` == 1 | `49_adult_social_assistance` == 2) |>
  filter(`49_adult_child_services` == 1 | `49_adult_child_services` == 2) |> 
  filter(`49_adult_food_banks` == 1 | `49_adult_food_banks` ==  2) 
  
ses_adult_count <- youth |>
  group_by(`49_adult_social_assistance`, 
           `49_adult_child_services`,
           `49_adult_food_banks`) |>
  summarize(count = n())


total_respondant_to_Q49 <- ses_adult_count |>
  ungroup() |>
  summarize(sum = sum(count))

ses_adult_count <- ses_adult_count |>
  ungroup() |>
  mutate(percentage_proportion = count / pull(total_respondant_to_Q49, 1) * 100)
ses_adult_count
# combine selections into one single column:
ses_adult_count <- ses_adult_count |>
  mutate(selection = paste(`49_adult_social_assistance`, 
                           `49_adult_child_services`, 
                           `49_adult_food_banks`)) |>
  mutate(selection = as_factor(selection))
write_csv(ses_adult_count, "outputs/tables/week-03-count-adult-ses-indicator-responses.csv")

glimpse(ses_adult_count)

# distribution
ses_adult_count |>
  mutate(id = row_number()) |>
  ggplot(aes(y = selection, x = count, fill = selection)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 0, hjust = .5)) +
  labs(title = "Distribution of Responses to Question 49 as \n SES indicators since turning 18 (age 19-30)",
       fill = "Response Selection",
       caption = "Response Selection Encoding for 3 questions: \n
       (a) Currently receiving social assistance (e.g., Welfare or disability support) \n
       (b) Had contact with child/family protective services \n
       (c) Experience using the food banks. \n
       1 = Yes, 2 = No",
       y = "Possible Combination of Response Selections")  + 
  theme(plot.caption.position = "plot",
        plot.caption = element_text(hjust = 0))
ggsave("outputs/figures/week-03-ses-indicator-adult-response-count.png")
## >>> Observations: 
# There seems to be an increase of respondents who are in a less fortunate 
# SES situation compared to when they were still teens.

#  Todo: Index out those who are > more than two 'Yes' responses to the 3 question and then 
#  look at their current income and their mentor experience:

#----------------------------------------------------------------------------------
# Of the subset of sample who answered at least 2 "Yes":

# Question 38 and Question 49 Re-coding from "1" --> "Yes", "2" --> "No"
youth_recoded <- youth |> 
  # Q38:
  mutate(`38_social_assistance` = recode(`38_social_assistance`, "1" = "Yes", "2" = "No"),
         `38_work_to_support_family` = recode(`38_work_to_support_family`, "1" = "Yes", "2" = "No"),
         `38_food_bank_use` = recode(`38_food_bank_use`, "1" = "Yes", "2" = "No")) |>
  # Q49:
  mutate(`49_adult_social_assistance` = recode(`49_adult_social_assistance`, "1" = "Yes", "2" = "No"),
         `49_adult_child_services` = recode(`49_adult_child_services`, "1" = "Yes", "2" = "No"),
         `49_adult_food_banks` = recode(`49_adult_food_banks`, "1" = "Yes", "2" = "No"))

two_or_more_yes_Q38 <- youth_recoded |> # Include only "Yes" and "No" Response for Q38
  filter(((`38_social_assistance` == "Yes") & (`38_work_to_support_family` == "Yes")) |
         ((`38_social_assistance` == "Yes") & (`38_food_bank_use` == "Yes")) |
         ((`38_work_to_support_family` == "Yes") & (`38_food_bank_use` == "Yes")) |
         ((`38_social_assistance` == "Yes") & (`38_work_to_support_family` == "Yes") & (`38_food_bank_use` == "Yes")))

two_or_more_yes_Q49 <- youth_recoded |> # Include only "Yes" and "No" Response and Q49
  filter(((`49_adult_social_assistance` == "Yes") & (`49_adult_child_services` == "Yes")) |
           ((`49_adult_social_assistance` == "Yes") & (`49_adult_food_banks` == "Yes")) |
           ((`49_adult_child_services` == "Yes") & (`49_adult_food_banks` == "Yes")) |
           ((`49_adult_social_assistance` == "Yes") & (`49_adult_child_services` == "Yes") & (`49_adult_food_banks` == "Yes"))) 

# --------------------- `18_early_mentor`, `19_teen_mentor`---------------------
## -------------- Barplot: 38_social_assistance & 18_early_mentor --------------
two_or_more_yes_Q38 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `38_social_assistance`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Family received social \n assistance in teen",
       title = "Comparing proportion of teens with social assistance with / without mentors as kids")

ggsave("outputs/figures/week-03-ses-teen-social-assistance-proportion-(early-mentor).png")


## Barplot: 38_work_to_support_family & 18_early_mentor
two_or_more_yes_Q38 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `38_work_to_support_family`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Worked to support family \n in teen",
       title = "Comparing proportion of teens who had to work to support family \n with / without mentors as kids")

ggsave("outputs/figures/week-03-ses-teen-worked-proportion-(early-mentor).png")

## Barplot: 38_food_bank_use & 18_early_mentor
two_or_more_yes_Q38 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `38_food_bank_use`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Food bank use in teen",
       title = "Comparing proportion of teens who had used the food bank \n with / without mentors as kids")

ggsave("outputs/figures/week-03-ses-teen-food-banks-proportion-(early-mentor).png")

## -------------- Barplot: 38_social_assistance & 19_teen_mentor --------------
two_or_more_yes_Q38 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `38_social_assistance`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 12-18 (teen)", fill = "Family received social \n assistance in teen",
       title = "Comparing proportion of teens with social assistance with / without mentors as teens-(teen-mentor)")

ggsave("outputs/figures/week-03-ses-teen-social-assistance-proportion.png")

## Barplot: 38_work_to_support_family & 19_teen_mentor
two_or_more_yes_Q38 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `38_work_to_support_family`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from 12-18 (teen)", fill = "Worked to support family \n in teen",
       title = "Comparing proportion of teens who had to work to support family \n with / without mentors as teens")
ggsave("outputs/figures/week-03-ses-teen-worked-proportion-(teen-mentor).png")

## Barplot: 38_food_bank_use & 19_teen_mentor
two_or_more_yes_Q38 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `38_food_bank_use`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from 12-18 (teen)", fill = "Food bank use in teen", 
       title = "Comparing proportion of teens who had used the food bank \n with / without mentors as teens")
ggsave("outputs/figures/week-03-ses-teen-food-banks-proportion-(teen-mentor).png")

# Q49: -----------------------------------------------------------------------------
# `49_adult_social_assistance`, `49_adult_child_services`, `49_adult_food_banks`
## --------------Barplot: 49_adult_social_assistance & 18_early_mentor--------------
two_or_more_yes_Q49 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `49_adult_social_assistance`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Family received social \n assistance in teen",
       title = "Comparing proportion of respondents with current social assistance \n with / without mentors as kids")
ggsave("outputs/figures/week-03-ses-adult-social-assistance-proportion-(early-mentor).png")


## Barplot: 49_adult_child_services & 18_early_mentor
two_or_more_yes_Q49 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `49_adult_child_services`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Experience with child \n services as adults",
       title = "Comparing proportion of respondents who had experience with \n child services as adults with / without mentors as kids")

ggsave("outputs/figures/week-03-ses-adult-child-service-proportion-(early-mentor).png")

## Barplot: 49_adult_food_banks & 18_early_mentor
two_or_more_yes_Q49 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), fill = `49_adult_food_banks`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 6-11", fill = "Food bank use in teen",
       title = "Comparing proportion of respondents who had used the food bank as adults \n with / without mentors as kids")
ggsave("outputs/figures/week-03-ses-adult-food-banks-proportion-(early-mentor).png")

## -------------- Barplot: 49_adult_social_assistance & 19_teen_mentor--------------
two_or_more_yes_Q49 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `49_adult_social_assistance`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from age 12-18 (teen)", fill = "Social assistance as adults",
       title = "Comparing proportion of respondents with social assistance as \n adults with / without mentors as teens")
ggsave("outputs/figures/week-03-ses-adult-social-assistance-proportion-(teen-mentor).png")

## Barplot: 49_adult_child_services & 19_teen_mentor
two_or_more_yes_Q49 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `49_adult_child_services`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from 12-18 (teen)", fill = "Experience with child \n services as adults",
       title = "Comparing proportion of respondents who had experience with \n child services as adults with / without mentors as teens")
ggsave("outputs/figures/week-03-ses-adult-child-service-proportion-(teen-mentor).png")

## Barplot: 49_adult_food_banks & 19_teen_mentor
two_or_more_yes_Q49 |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), fill = `49_adult_food_banks`)) +
  geom_bar(position = "fill") + 
  labs(x = "Mentorship experience from 12-18 (teen)", fill = "Food bank use as adults", 
       title = "Comparing proportion of respondents who had used the food \n bank as adults with / without mentors as teens")
ggsave("outputs/figures/week-03-ses-adult-food-banks-proportion-(teen-mentor).png")


# -----------------------------Income----------------------------
# Of these individuals who answered at least 2 "Yes" to the three questions, we also
#  look at their current income situation while comparing their mentor experience:
# `15_yearly_income`
## -------------- Barplot: 38_social_assistance & 18_early_mentor& 15_yearly_income --------------
two_or_more_yes_Q38 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `15_yearly_income`, fill = `38_social_assistance`)) +
  geom_boxplot() 
# Observation: Extreme income earner outlier shown in boxplot 
two_or_more_yes_Q38 |>
  group_by(`18_early_mentor`, `38_social_assistance`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = (`median_yearly_income`), fill = `38_social_assistance`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had early mentor \n experience (from age 6-11) and received social assistance as teen (from age 12-18)", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Received social assistance \n as teens (from age 12-18)")

ggsave("outputs/figures/week-03-ses-income-teen-social-assistance-(early-mentor).png")

## Barplot: 38_work_to_support_family & 18_early_mentor
two_or_more_yes_Q38 |>
  group_by(`18_early_mentor`, `38_work_to_support_family`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `median_yearly_income`, fill = `38_work_to_support_family`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had early mentor \n experience (from age 6-11) and worked to support family as teen (from age 12-18)", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Worked to support \n family as teens (from age 12-18)")
ggsave("outputs/figures/week-03-ses-income-teen-worked-(early-mentor).png")

## Barplot: 38_food_bank_use & 18_early_mentor
two_or_more_yes_Q38 |>
  group_by(`18_early_mentor`, `38_food_bank_use`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `median_yearly_income`, fill = `38_food_bank_use`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had early mentor \n experience (from age 6-11) and used food banks services as teen (from age 12-18)", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Used food bank ervices \n as teen (from age 12-18)")
ggsave("outputs/figures/week-03-ses-income-teen-food-banks-(early-mentor).png")
## -------------- Barplot: 38_social_assistance & 19_teen_mentor --------------
two_or_more_yes_Q38 |>
  group_by(`19_teen_mentor`, `38_social_assistance`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = (`median_yearly_income`), fill = `38_social_assistance`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and received social assistance as teen (from age 12-18)", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Family received social \n assistance as teen (from age 12-18)")
ggsave("outputs/figures/week-03-ses-income-teen-social-assistance-(teen-mentor).png")

## Barplot: 38_work_to_support_family & 18_early_mentor
two_or_more_yes_Q38 |>
  group_by(`19_teen_mentor`, `38_work_to_support_family`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = `median_yearly_income`, fill = `38_work_to_support_family`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and worked to support family as teen (from age 12-18)", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Worked to support family \n as teen (from age 12-18)")
ggsave("outputs/figures/week-03-ses-income-teen-worked-(teen-mentor).png")
## Barplot: 38_food_bank_use & 19_teen_mentor
two_or_more_yes_Q38 |>
  group_by(`19_teen_mentor`, `38_food_bank_use`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = `median_yearly_income`, fill = `38_food_bank_use`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individualswho had mentor experience \n as teen (from age 12-18) and used food banks services as teen (from age 12-18)", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Used food bank ervices \n as teen (from age 12-18)")
ggsave("outputs/figures/week-03-ses-income-teen-food-banks-(teen-mentor).png")
## ------------------ Income Q39 ------------------
## -------------- 15_yearly_income --------------
# `49_adult_social_assistance`, `49_adult_child_services`, `49_adult_food_banks`

two_or_more_yes_Q49 |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `15_yearly_income`, fill = `49_adult_social_assistance`)) +
  geom_boxplot() 
# Observation: Extreme income earner outlier shown in boxplot 
two_or_more_yes_Q49 |>
  group_by(`18_early_mentor`, `49_adult_social_assistance`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = (`median_yearly_income`), fill = `49_adult_social_assistance`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and currently receiving social assistance as adult", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Currently receiving social \n assistance as adult")
ggsave("outputs/figures/week-03-ses-income-adult-social-assistance-proportion-(early-mentor).png")

## Barplot: 49_adult_child_services & 18_early_mentor
two_or_more_yes_Q49 |>
  group_by(`18_early_mentor`, `49_adult_child_services`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>  
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `median_yearly_income`, fill = `49_adult_child_services`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and have contacted with child services as adult", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Have contacted with \n child services as adult")
ggsave("outputs/figures/week-03-ses-income-adult-child-service-proportion-(early-mentor).png")
## Barplot: 38_food_bank_use & 18_early_mentor
two_or_more_yes_Q49 |>
  group_by(`18_early_mentor`, `49_adult_food_banks`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(`18_early_mentor` == "No" | `18_early_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`18_early_mentor`), y = `median_yearly_income`, fill = `49_adult_food_banks`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and used the food banks as adult", 
       x = "Early experience with mentorship (age 6-11)", 
       y = "Estimated yearly median income",  fill = "Food bank uses as adult")
ggsave("outputs/figures/week-03-ses-income-adult-food-banks-proportion-(early-mentor).png")
## -------------- Barplot: 38_social_assistance & 19_teen_mentor --------------
## Barplot: 38_food_bank_use & 19_teen_mentor
two_or_more_yes_Q49 |>
  group_by(`19_teen_mentor`, `49_adult_social_assistance`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = `median_yearly_income`, fill = `49_adult_social_assistance`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and currently receiving social assistance as adult", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Currently receiving social \n assistance as adult")
ggsave("outputs/figures/week-03-ses-income-adult-social-assistance-proportion-(teen-mentor).png")
## Barplot: 38_work_to_support_family & 18_early_mentor
two_or_more_yes_Q49 |>
  group_by(`19_teen_mentor`, `49_adult_child_services`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = `median_yearly_income`, fill = `49_adult_child_services`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and have contacted with child services as adult", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Have contacted with \n child services as adult")
ggsave("outputs/figures/week-03-ses-income-adult-child-service-proportion-(teen-mentor).png")

## Barplot: 38_food_bank_use & 18_early_mentor
two_or_more_yes_Q49 |>
  group_by(`19_teen_mentor`, `49_adult_food_banks`) |>
  summarize(median_yearly_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(`19_teen_mentor` == "No" | `19_teen_mentor` == "Yes") |>
  ggplot(aes(x = as_factor(`19_teen_mentor`), y = `median_yearly_income`, fill = `49_adult_food_banks`)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparing current income earnings among individuals who had mentor experience \n as teen (from age 12-18) and used the food banks as adult", 
       x = "Teen experience with mentorship (age 12-18)", 
       y = "Estimated yearly median income",  fill = "Food bank uses as adult")
ggsave("outputs/figures/week-03-ses-income-adult-food-banks-proportion-(teen-mentor).png")
