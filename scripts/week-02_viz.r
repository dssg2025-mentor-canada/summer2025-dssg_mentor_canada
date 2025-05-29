library(tidyverse)
library(tidymodels)

youth <- read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')
head(youth)

youth <- youth |>
          mutate(across(2_province:10_caregiver, as_factor)) |>
          mutate(across(12_highschool_ged:14_employment))


