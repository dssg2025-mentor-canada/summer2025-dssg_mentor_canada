library(tidyverse)
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
  filter(`45_mental_health_rating` %in% c("Good", "Very good", "Excellent", "Fair", "Poor")) |>
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

