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

# Estimated yearly income distribution (15_yearly_income)
median_income <- median(youth$`15_yearly_income`, na.rm = TRUE)
mean_income <- mean(youth$`15_yearly_income`, na.rm = TRUE)

youth |>
  ggplot(aes(x = `15_yearly_income`)) +
  geom_histogram(color = "white", fill = "yellowgreen") +
  geom_vline(aes(xintercept = mean(`15_yearly_income`)), linetype="dashed", size = 1, colour = "darkred") +
  scale_y_log10() +
  annotate("text", x = 17000000, y = 100, label = paste("Median yearly income = $", round(median_income, 2))) +
  labs(title = "Distribution of Survey Participant Estimated Yearly Income", x = "Estimated Yearly Income ($)",
       y = "Frequenct Count") +
  theme(axis.title = element_text(size = 13),
        axis.text = element_text(size = 11),
        plot.title = element_text(size = 15))

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
