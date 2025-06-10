# imports
import pandas as pd

# load in data
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/youth_tidy.csv')

# counts
data['32_mentor_helpfulness'] = data['32_mentor_helpfulness'].replace({1.0: 'Very helpful',
                                       2.0: 'Fairly helpful',
                                       3.0: 'Somewhat helpful',
                                       4.0: 'Not that helpful',
                                       5.0: 'Not helpful at all',
                                       6.0: 'Unsure',
                                       7.0: 'Prefer not to say'}).copy()

print(data['32_mentor_helpfulness'].value_counts())

# binning into helpful (1) /not helpful (0)
def bin_helpfulness(x):
    if x in ['Very helpful', 'Fairly helpful', 'Somewhat helpful']:
        return 1
    elif x in ['Not that helpful', 'Not helpful at all']:
        return 0
    else:
        return None # exclude
    

data['mentor_helpful_bin'] = data['32_mentor_helpfulness'].apply(bin_helpfulness)
data = data.dropna(subset=['mentor_helpful_bin'])

print(data['mentor_helpful_bin'].value_counts())


# possible variables to explore
# Q29, transitions
# Q26, match similarity
# Q22, mentor age relative to mentee age

table = pd.crosstab(data['mentor_helpful_bin'], data['26_religion_similar'])
print(table)

