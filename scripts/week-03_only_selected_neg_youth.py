import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


path = "../../dssg-2025-mentor-canada/Data/neg_life_events_youth.csv"
data = pd.read_csv(path)

# only_selected_df = data[data['num_neg_life_events_youth'] != 0]

# counts = only_selected_df['num_neg_life_events_youth'].value_counts().sort_index()

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


ax = pivot_df.plot(
    kind="bar", 
    stacked=True, 
    figsize=(10, 6)
)

for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize=8)

plt.xlabel("Number of Negative Life Events Experienced")
plt.ylabel("Count of People")
plt.title("Stacked Breakdown by Type of Negative Life Event")
plt.legend(title="Event Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()

# plt.figure(figsize=(8,5))
# plt.bar(counts.index, counts.values)

# ax = plt.gca()  # get current axes
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# plt.xlabel("Number of Negative Life Events Experienced as Youth")
# plt.ylabel("Count of People that Experienced \nThis Number of Events")
# plt.title("Distribution of Negative Life Events Experienced During Youth (12-18) \n(Excludes Selections with None)")

# plt.show()