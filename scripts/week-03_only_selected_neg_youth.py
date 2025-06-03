import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


path = "../../dssg-2025-mentor-canada/Data/neg_life_events_youth.csv"
data = pd.read_csv(path)

# plot i: looking at composition of selections for 1 option selected
only_selected_df = data[data['num_neg_life_events_youth'] == 1]

subgroup_cols = [
    "38_food_bank_use", 
    "38_social_assistance", 
    "38_work_to_support_family"
]

melted = only_selected_df.melt(
    id_vars="num_neg_life_events_youth",
    value_vars=subgroup_cols,
    var_name="event_type",
    value_name="response"
)

# Keep only rows where the person said "yes" (1)
melted_yes = melted[melted["response"] == 1]

grouped = (
    melted_yes
    .groupby(["num_neg_life_events_youth", "event_type"])
    .size()
    .reset_index(name="count")
)

pivot_df = grouped.pivot(index="num_neg_life_events_youth", columns="event_type", values="count").fillna(0)

# plot ii: looking at composition of selections for 2 option selected
selected_2_df = data[data['num_neg_life_events_youth'] == 2]

# create counts for the different combinations of two selections: 3 possible combos
def combos_of_2(df):

    food_social = 0
    food_work = 0
    work_social = 0

    for _, row in df.iterrows():
        if row["38_food_bank_use"] == 1 & row["38_social_assistance"] == 1:
            food_social += 1
        if row["38_food_bank_use"] == 1 & row["38_work_to_support_family"] == 1:
            food_work += 1
        if row["38_social_assistance"] == 1 & row["38_work_to_support_family"] == 1:
            work_social += 1
    
    return pd.DataFrame([{
        "Food Bank Use + Social Assistance": food_social,
        "Food Bank Use + Needed to Work": food_work,
        "Needed to Work + Social Assistance": work_social
        }])


select_2 = combos_of_2(selected_2_df)

# set up the two plots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# 1st stacked bar chart
pivot_df.plot(kind="bar", stacked=True, ax=ax1)
ax1.set_title("Group 1: Youth with 1 Negative Event")
ax1.set_xlabel("Event Count")
ax1.set_ylabel("Count of People")
ax1.legend(title="Event Type", loc='upper right')

# 2nd stacked bar chart
select_2.plot(kind="bar", stacked=True, ax=ax2)
ax2.set_title("Group 2: Youth with 2 Negative Events")
ax2.set_xlabel("Event Count")
ax2.legend(title="Event Type", loc='upper right')

# Optional: Add bar labels
for ax in (ax1, ax2):
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

plt.tight_layout()
plt.show()
