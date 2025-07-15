library(tidyverse)
library(ggpubr)
youth <- read_rds('../../dssg-2025-mentor-canada/Data/processed_youth_factorized.RDS')
glimpse(youth)

# Age distribution (1_age)
mean_age <- mean(youth$`1_age`)
youth |>
  ggplot(aes(x = `1_age`)) +
  geom_histogram(binwidth = 1, position = "identity",  color = "white", fill = "yellowgreen") +
  geom_vline(aes(xintercept = mean(`1_age`)), linetype="dashed", size = 1, colour = "darkred") +
  annotate("text", x =  mean_age + 1.5, y = 100, label = paste("Mean age =", round(mean_age, 2))) +
  labs(title = "Distribution of Survey Participant Age", x = "Age", y = "Frequency Count") +
  theme(axis.title = element_text(size = 13),
        axis.text = element_text(size = 14))
ggsave("outputs/01_age_distribution.png")

# Estimated yearly income distribution (15_yearly_income)
median_income <- median(youth$`15_yearly_income`, na.rm = TRUE)
mean_income <- mean(youth$`15_yearly_income`, na.rm = TRUE)

youth |>
  ggplot(aes(x = `15_yearly_income`)) +
  geom_histogram(color = "white", fill = "yellowgreen") +
  geom_vline(aes(xintercept = median(`15_yearly_income`)), linetype="dashed", size = 1, colour = "darkred") +
  scale_y_log10() +
  annotate("text", x = 20000000, y = 100, label = paste("Median yearly income = $", round(median_income, 2))) +
  labs(title = "Distribution of Survey Participant Estimated Yearly Income", x = "Estimated Yearly Income ($)",
       y = "Frequenct Count") +
  theme(axis.title = element_text(size = 13),
        axis.text = element_text(size = 11),
        plot.title = element_text(size = 15))
ggsave("outputs/15_yearly_income_distribution.png")

# Data source: survey participant are recruited from which provinces in Canada? (`2_province`)

youth |>
  group_by(`2_province`) |>
  summarize(count = n()) |>
  ggplot(aes(y = fct_reorder(`2_province`, count, .desc = FALSE),
             x = count, fill = `2_province`)) +
  geom_bar(stat = "identity") + 
  labs(title = "Province Location of Survey Respondants", y = "Province (Canada)", fill = "Province (Canada)") +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) 
ggsave("outputs/02_province_barplot.png")

# Trans identity (`6_gender_identity`) - early mentor exposure (6-11, `18_early_mentor`) - Mental health output: `45_mental_health_rating`
# Teen mentor exposure (12-18, `19_teen_mentor`)
# - Income output: `14_employment`, `15_yearly_income`

youth |>
  ggplot(aes(y = `15_yearly_income`, x = fct_reorder(`7_trans_ident`, `15_yearly_income`, .desc = FALSE), fill = `7_trans_ident`)) +
  geom_bar(stat = "identity") +
  labs(y = "Estimated Yearly Income", x = "Transgender Identity", 
       title = "Estimated Yearly Income of Individuals with or without Transgender Experience",
       fill = "Transgender Identity")

ggsave("outputs/07_trans_identity-income.png")

# Mental health
mental_health_order <- c("Excellent", "Very good", "Good", "Fair", "Poor")
youth |>
  filter(`18_early_mentor` %in% c("Yes", "No")) |>
  filter(`45_mental_health_rating` %in% c("Good", "Very good", "Excellent", "Poor")) |>
  filter(!is.na(`18_early_mentor`)) |>
  drop_na(`45_mental_health_rating`) |>
  ggplot(aes(x = `18_early_mentor`, fill = factor(`45_mental_health_rating`, levels = mental_health_order))) +
  geom_bar(position = "fill") +
  labs(x = "Early life mentor experience (age 6-11 years old)", fill = "Current Mental Health Rating", 
       title = "Mental Health Rating Comparing between Mentorship Participation from Age 6-11")

ggsave("outputs/18_early_mentor-mental_health.png")

youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`45_mental_health_rating` %in% c("Good", "Very good", "Excellent", "Fair", "Poor")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  drop_na(`45_mental_health_rating`) |>
  ggplot(aes(x = `19_teen_mentor`, fill = factor(`45_mental_health_rating`, levels = mental_health_order))) +
  geom_bar(position = "fill") +
  labs(x = "Early life mentor experience (age 6-11 years old)", fill = "Current Mental Health Rating", 
       title = "Mental Health Rating Comparing between Mentorship Participation from Age 12-18")

ggsave("outputs/19_teen_mentor-mental_health.png")

# >>--------- update ---------<<
# -- Income--
# barplot - average
youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  group_by(`19_teen_mentor`)|>
  summarize(average_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, y = average_income, fill = `19_teen_mentor`)) +
  geom_bar(stat = 'identity') +
  labs(x = "Teen mentor experience", y = "Average current income", fill = "Teen Mentor", 
       title = "Average income of respondent with/without mentor from Age 12-18")+
         scale_fill_brewer(palette = 'Accent')

  
 # Remove income outliers:
youth |>
  filter(`18_early_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
  group_by(`18_early_mentor`)|>
  summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`18_early_mentor`)) |>
  ggplot(aes(x = `18_early_mentor`, y = median_income, fill = `18_early_mentor`)) +
  geom_bar(stat = 'identity') +
  labs(x = "Teen mentor experience", y = "Median current income ($ per year)", fill = "Childhood mentor \n experience", 
       title = "Average income of respondents with/without \n mentor from Age 6-11") +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
  group_by(`19_teen_mentor`)|>
  summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, y = median_income, fill = `19_teen_mentor`)) +
  geom_bar(stat = 'identity') +
  labs(x = "Teen mentor experience", y = "Median current income ($ per year)", fill = "Teen mentor \n expererience", 
       title = "Average income of respondents with/without \n mentor from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


# boxplot - all (Shows extreme outlier earner)
youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, y = `15_yearly_income`, fill = `19_teen_mentor`)) +
  geom_boxplot() + # replace with  geom_boxplot(outlier.shape = NA) to remove outlier
 # coord_cartesian(ylim=c(0, 130000)) + # set y range limit
  labs(x = "Teen mentor experience", y = "Average current income", fill = "Early mentor \n expereince", 
       title = "Average income of respondent with/without mentor from Age 6-12") +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`18_early_mentor` %in% c("Yes", "No")) |>
  filter(!is.na(`18_early_mentor`)) |>
  ggplot(aes(x = `18_early_mentor`, y = `15_yearly_income`, fill = `18_early_mentor`)) +
  geom_boxplot() +  # replace with  geom_boxplot(outlier.shape = NA) to remove outlier
 # coord_cartesian(ylim=c(0, 130000)) + # set y range limit
  labs(x = "Teen mentor experience", y = "Average current income", fill = "Early mentor \n expereince", 
       title = "Average income of respondent with/without mentor from Age 6-12")+
         scale_fill_brewer(palette = 'Accent')


# -- GED --
youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `12_highschool_ged`  %in% c("Yes", "No")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, fill = factor(`12_highschool_ged`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Teen mentor experience", fill = "High school \n diploma", 
       title = "High school GED with/without Mentorship \n Participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`18_early_mentor` %in% c("Yes", "No"),
          `12_highschool_ged`  %in% c("Yes", "No")) |>
  filter(!is.na(`18_early_mentor`)) |>
  ggplot(aes(x = `18_early_mentor`, fill = factor(`12_highschool_ged`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Early mentor experience", fill = "High school diploma", 
       title = "High school GED with/without Mentorship \n Participation from Age 6-11") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `13_further_edu`  %in% c("Yes", "No")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, fill = factor(`13_further_edu`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Teen mentor experience", fill = "Post-secondary \n Education", 
       title = "Post-secondary education attainment with/without \n Mentorship Participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`18_early_mentor` %in% c("Yes", "No"),
          `13_further_edu`  %in% c("Yes", "No")) |>
  filter(!is.na(`18_early_mentor`)) |>
  ggplot(aes(x = `18_early_mentor`, fill = factor(`13_further_edu`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Early mentor experience", fill = "Post-secondary \n Education", 
       title = "Post-secondary education attainment with/without \n  Mentorship Participation from Age 6-11") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `38_work_to_support_family`  %in% c("1", "2")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, fill = factor(`38_work_to_support_family`, levels = c("1", "2")))) +
  geom_bar(position = "fill") +
  labs(x = "Teen mentor experience", fill = "Worked to support \n family in youth \n 1 = Yes \n 2 = No", 
       title = "Whether worked to support family in youth \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')

# 38_work_to_support_family

youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  group_by(`19_teen_mentor`) |>
  summarize(average_problem_solve = mean(`46_problem_solve`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, y = average_problem_solve, fill = `19_teen_mentor`)) +
  geom_bar(stat = 'identity') +
  labs(x = "Teen mentor experience", y = "Average rating of confidence", fill = "Received mentorship \n as teen", 
       title = "Average rating of confidence in problem solving as adult  \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No")) |>
  group_by(`19_teen_mentor`) |>
  summarize(average_optimism = mean(`46_optimism`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, y = average_optimism, fill = `19_teen_mentor`)) +
  geom_bar(stat = 'identity') +
  labs(x = "Teen mentor experience", fill = "Received mentorship \n as teen", 
       title = "Average rating of optimism & Youth mentorship \n participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `38_work_to_support_family`  %in% c("1", "2")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `19_teen_mentor`, fill = factor(`38_work_to_support_family`, levels = c("1", "2")))) +
  geom_bar(position = "dodge") +
  labs(x = "Teen mentor experience", fill = "Worked to support \n family in youth \n 1 = Yes \n 2 = No", 
       title = "Whether worked to support family in youth \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


# --SES--
youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `38_food_bank_use`  %in% c("1", "2")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `38_food_bank_use`, fill = factor(`19_teen_mentor`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Use of foodbank in youth \n 1 = Yes | 2 = No", fill = "Teen mentor \n experience", 
       title = "Foodbank use in youth \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `38_work_to_support_family`  %in% c("1", "2")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `38_work_to_support_family`, fill = factor(`19_teen_mentor`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Use of foodbank in youth \n 1 = Yes | 2 = No", fill = "Teen mentor \n experience", 
       title = "Foodbank use in youth \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')



youth |>
  filter(`19_teen_mentor` %in% c("Yes", "No"),
          `38_social_assistance`  %in% c("1", "2")) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `38_social_assistance`, fill = factor(`19_teen_mentor`, levels = c("Yes", "No")))) +
  geom_bar(position = "fill") +
  labs(x = "Use of foodbank in youth \n 1 = Yes | 2 = No", fill = "Teen mentor \n experience", 
       title = "Foodbank use in youth \n & Youth mentorship participation from Age 12-18") +
          theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


#--SES & income--

welfare_income_plot <- youth |>
  filter(`38_social_assistance` %in% c("1", "2"),
         `18_early_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
  group_by(`18_early_mentor`, `38_social_assistance`)|>
  summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`18_early_mentor`)) |>
  ggplot(aes(x = `38_social_assistance`, y = median_income, fill = `18_early_mentor`)) +
  geom_bar(stat = 'identity', position = 'dodge') +
  scale_x_discrete(labels=c('Yes', 'No')) +
          labs(x = "Received fianancial assistance during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
       title = "Income of respondents who \nreceived financial support in youth") +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11))  +
         scale_fill_brewer(palette = 'Accent')

worked_income_plot <- youth |>
  filter(`38_care_for_family` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
  group_by(`19_teen_mentor`, `38_care_for_family`)|>
  summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
  ggplot(aes(x = `38_care_for_family`, y = median_income)) +
  geom_bar(stat = 'identity', position = 'dodge', aes(fill = `19_teen_mentor`)) +
  scale_x_discrete(labels=c('Yes', 'No')) +
        labs(x = "Worked to support family during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
        title = "Income of respondents who \nworked to support family in youth") +
  theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')


food_bank_income_plot <- youth |>
  filter(`38_food_bank_use` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
  group_by(`19_teen_mentor`, `38_food_bank_use`)|>
  summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
        ungroup() |>
  filter(!is.na(`19_teen_mentor`)) |>
        ggplot(aes(x = `38_food_bank_use`, y = `median_income`, fill = `38_food_bank_use`)) +
  geom_bar(stat = 'identity', position = 'dodge', aes(fill = `19_teen_mentor`)) +
          labs(x = "Foodbank uses during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
       title = "Income of respondents who used \nfoodbanks in youth") +
        scale_x_discrete(labels=c('Yes', 'No')) +
        theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')

ggarrange(welfare_income_plot, worked_income_plot,food_bank_income_plot, 
        labels = c("A", "B", "C"), ncol = 3, nrow = 1, common.legend = TRUE, legend = "top")
#----- boxplots:

youth |>
  filter(`38_care_for_family` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income

  filter(!is.na(`19_teen_mentor`)) |>
        ggplot(aes(x = `38_care_for_family`, y = `15_yearly_income`, fill = `19_teen_mentor`)) +
        geom_boxplot() +
        labs(x = "Worked to support family during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
        title = "Median income of respondents \nwho worked to support \nfamily during adolescence") +
        scale_x_discrete(labels=c('Yes', 'No')) +
        theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')

youth |>
  filter(`38_food_bank_use` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
#   group_by(`19_teen_mentor`, `38_food_bank_use`)|>
#   summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`19_teen_mentor`)) |>
        ggplot(aes(x = `38_food_bank_use`, y = `15_yearly_income`, fill = `19_teen_mentor`)) +
        geom_boxplot() +
          labs(x = "Foodbank uses during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
       title = "Median income of respondents who used \nfoodbanks during adolescence") +
        scale_x_discrete(labels=c('Yes', 'No')) +
        theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')

#   ggplot(aes(x = `38_food_bank_use`, y = median_income)) +
#   geom_bar(stat = 'identity', position = 'dodge', aes(fill = `19_teen_mentor`)) +
#   
youth |>
  filter(`38_food_bank_use` %in% c("1", "2"),
         `18_early_mentor` %in% c("Yes", "No")) |>
  filter(`15_yearly_income` < 130000) |> # filter out extreme outlier earners > $130,000 yearly income
#   group_by(`19_teen_mentor`, `38_food_bank_use`)|>
#   summarize(median_income = median(`15_yearly_income`, na.rm = TRUE)) |>
  filter(!is.na(`18_early_mentor`)) |>
        ggplot(aes(x = `38_food_bank_use`, y = `15_yearly_income`, fill = `18_early_mentor`)) +
        geom_boxplot() +
          labs(x = "Foodbank uses during youth", y = "Median current income ($ per year)", fill = "Teen mentor \n experience", 
       title = "Median income of respondents who used \nfoodbanks during adolescence") +
        scale_x_discrete(labels=c('Yes', 'No')) +
        theme(axis.text = element_text(size = 11),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14),
        legend.text = element_text(size = 11)) +
         scale_fill_brewer(palette = 'Accent')

# Summary - those who were low SES & proportion of people had mentor
# Social welfare:
youth |>
  filter(`38_social_assistance` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_social_assistance`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_social_assistance` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
#  Of those who didn't receive social assistance in you (higher ses), 54.0% had mentor compared to No mentor.

youth |>
  filter(`38_social_assistance` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_social_assistance`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_social_assistance` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
#  Of those who didn't receive social assistance in you (higher ses), 44.4% had mentor compared to No mentor.

# Worked for family:
youth |>
  filter(`38_care_for_family` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_care_for_family`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_care_for_family` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# 51.6% of those who had to work(lower ses) who had mentor compared to No mentor.

youth |>
  filter(`38_care_for_family` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_care_for_family`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_care_for_family` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# 44.2% of those who didn't have to work (higher ses) who had mentor compared to No mentor.

# Foodbank:
youth |>
  filter(`38_food_bank_use` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_food_bank_use`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_food_bank_use` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
#  Of those who had to use foodbankin you (higher ses), 45% had mentor compared to No mentor.

youth |>
  filter(`38_food_bank_use` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_food_bank_use`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_food_bank_use` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
#  Of those who did not have to use foodbank in you (higher ses), 45% had mentor compared to No mentor.

# --- Other intereseting observations:----
# 1. School absence (social withdrawl) - no difference in mentor exposure between Yes/No school absence
youth |>
  filter(`38_school_absence` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_school_absence`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_school_absence` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
#  Of those who had to use foodbankin you (higher ses), 45% had mentor compared to No mentor.

youth |>
  filter(`38_school_absence` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_school_absence`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_school_absence` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 

# 2. Youth in care - Less mentor exposure between being / not being a youth in care
youth |>
  filter(`38_youth_in_care` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_youth_in_care`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_youth_in_care` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# Of those who were youth in care, 49.2% had mentor in youth
youth |>
  filter(`38_youth_in_care` %in% c("1", "2"),
         `19_teen_mentor` %in% c("Yes", "No")) |>
          group_by(`38_youth_in_care`,`19_teen_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_youth_in_care` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# Of those who were youth in care, 44.2% had mentor in youth

# 2.1 Early mentor / youth in care
youth |>
  filter(`38_youth_in_care` %in% c("1", "2"),
         `18_early_mentor` %in% c("Yes", "No")) |>
          group_by(`38_youth_in_care`, `18_early_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_youth_in_care` == 1) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# Of those who were youth in care, 49.2% had mentor in youth
youth |>
  filter(`38_youth_in_care` %in% c("1", "2"),
         `18_early_mentor` %in% c("Yes", "No")) |>
          group_by(`38_youth_in_care`,`18_early_mentor` ) |> 
        summarize(count = n()) |>
        ungroup() |>
        filter(`38_youth_in_care` == 2) |>
        summarize(ses_worked_had_mentor_prop =  (count[1])/sum(count)) 
# Of those who were youth in care, 44.2% had mentor in youth

