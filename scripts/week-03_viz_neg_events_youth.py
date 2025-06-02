import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

path = "../../dssg-2025-mentor-canada/Data/neg_life_events_youth.csv"
data = pd.read_csv(path)

counts = data['num_neg_life_events_youth'].value_counts().sort_index()

plt.figure(figsize=(8,5))
plt.bar(counts.index, counts.values)

ax = plt.gca()  # get current axes
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.xlabel("Number of Negative Life Events Experienced as Youth")
plt.ylabel("Count of People that Experienced \nThis Number of Events")
plt.title("Distribution of Negative Life Events Experienced During Youth (12-18)")

plt.show()
