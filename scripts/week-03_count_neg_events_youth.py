import pandas as pd

def count_neg_youth(df):
    cols = ["38_social_assistance", "38_work_to_support_family", "38_food_bank_use"]
    df["num_neg_life_events_youth"] = df[cols].eq(1).sum(axis=1)

    return df[cols + ["num_neg_life_events_youth"]]


mentor_df = pd.read_csv("../../dssg-2025-mentor-canada/Data/youth_tidy.csv")

neg_events_youth_count = count_neg_youth(mentor_df)
neg_events_youth_count.to_csv("../../dssg-2025-mentor-canada/Data/neg_life_events_youth.csv", index=False)

print(neg_events_youth_count)