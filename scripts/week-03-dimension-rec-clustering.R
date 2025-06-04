library(tidyverse)
library(GGally)
# `38_social_assistance`, `38_work_to_support_family`, `38_food_bank_use`

youth <- read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')
glimpse(youth)
install.packages(c("Gifi", "psych", "cluster", "ggplot2", "factoextra"))
