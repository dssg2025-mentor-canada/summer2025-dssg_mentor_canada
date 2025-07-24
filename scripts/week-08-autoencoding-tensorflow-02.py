import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# import tensorflow as tf
# import tensorflow.keras.layers as layers
# import tensorflow.keras.models as models


# import data
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/faiss_tuned_knn_imputed_train.csv')
test_data = pd.read_csv('../../dssg-2025-mentor-canada/Data/faiss_tuned_knn_imputed_test.csv')


# column type definition
metric_vars = ['QS1_1_AGE', 'QS1_8_NEWCOMERYEAR', 'QS1_14_DISABIL', 'QS1_15_DISABIL', 
               'QS1_18_PARENTEDUC1', 'QS1_18_PARENTEDUC2',
               'QS1_20_HIGHSCHOOL', 'QS1_23_YEARCOMPLE', 'QS2_10_NUMBEROFME', 
               'QS4_15_TIMEIFFOR1', 'QS4_15_TIMEIFFOR2', 'QS4_15_TIMEIFFOR3', 
               'QS4_15_TIMEIFFOR4', 'QS4_12_TRUST1_1_1', 'QS4_12_TRUST1_2_2', 
               'QS4_12_TRUST1_3_3', 'QS4_12_TRUST1_4_4', 'QS4_12_TRUST1_5_5', 
               'QS4_14_FORMALVOL', 'QS1_28_EMPLOYMENT_calculated']
continuous_prefixes = ('QS1_1_', 'QS1_8_', 'QS1_14_', 'QS1_15_',
                       'QS1_20_', 'QS1_23_', 'QS2_10_', 'QS4_15_',
                       'QS4_12_', 'QS4_14_')

ordinal_numeric_vars = [
                        'QS2_19_DURATION_1', 'QS2_20_EXPERIENCE_1', 
                        'QS2_19_DURATION_2', 'QS2_20_EXPERIENCE_2', 
                        'QS2_19_DURATION_3', 'QS2_20_EXPERIENCE_3', 
                        'QS2_30_MATCHSIMILAR1_1_1', 'QS2_30_MATCHSIMILAR1_2_2', 
                        'QS2_30_MATCHSIMILAR1_3_3', 'QS2_30_MATCHSIMILAR1_4_4', 
                        'QS2_30_MATCHSIMILAR1_5_5', 'QS2_31_MENTORINGREL1_1_1', 
                        'QS2_31_MENTORINGREL1_2_2', 'QS2_31_MENTORINGREL1_3_3', 
                        'QS2_31_MENTORINGREL1_4_4', 'QS2_31_MENTORINGREL1_5_5', 
                        'QS2_32_MENTORINGENG1_1_1', 'QS2_32_MENTORINGENG1_2_2', 
                        'QS2_32_MENTORINGENG1_3_3', 'QS2_32_MENTORINGENG1_4_4', 
                        'QS2_32_MENTORINGENG1_5_5', 'QS2_32_MENTORINGENG1_6_6', 
                        'QS2_32_MENTORINGENG1_7_7', 'QS2_32_MENTORINGENG1_8_8', 
                        'QS2_32_MENTORINGENG1_9_9', 'QS2_32_MENTORINGENG1_10_10', 
                        'QS2_32_MENTORINGENG1_11_11', 'QS2_32_MENTORINGENG1_12_12', 
                        'QS2_32_MENTORINGENG1_13_13', 'QS2_32_MENTORINGENG1_14_14', 
                        'QS2_32_MENTORINGENG1_15_15', 'QS2_32_MENTORINGENG1_16_16', 
                        'QS2_32_MENTORINGENG1_17_17', 'QS2_32_MENTORINGENG1_18_18',
                        'QS2_32_MENTORINGENG1_19_19', 'QS2_32_MENTORINGENG1_20_20', 
                        'QS2_32_MENTORINGENG1_21_21', 'QS2_32_MENTORINGENG1_22_22', 
                        'QS2_35_SUPPORTSIMPO1_1_1', 'QS2_35_SUPPORTSIMPO1_2_2', 
                        'QS2_35_SUPPORTSIMPO1_3_3', 'QS2_35_SUPPORTSIMPO1_4_4', 
                        'QS2_35_SUPPORTSIMPO1_5_5', 'QS2_35_SUPPORTSIMPO1_6_6', 
                        'QS2_35_SUPPORTSIMPO1_7_7', 'QS2_35_SUPPORTSIMPO1_8_8', 
                        'QS2_35_SUPPORTSIMPO1_9_9', 'QS2_35_SUPPORTSIMPO1_10_10', 
                        'QS2_37_HELPFULNESS', 'QS3_1_GLOBALSELFWOR1_1_1', 
                        'QS3_1_GLOBALSELFWOR1_2_2', 'QS3_1_GLOBALSELFWOR1_3_3', 
                        'QS3_1_GLOBALSELFWOR1_4_4', 'QS3_1_GLOBALSELFWOR1_5_5', 
                        'QS3_1_GLOBALSELFWOR1_6_6', 'QS3_1_GLOBALSELFWOR1_7_7', 
                        'QS3_1_GLOBALSELFWOR1_8_8', 'QS3_5_SCHOOLCLIMATE1_1_1', 
                        'QS3_5_SCHOOLCLIMATE1_2_2', 'QS3_5_SCHOOLCLIMATE1_3_3', 
                        'QS3_5_SCHOOLCLIMATE1_4_4', 'QS3_5_SCHOOLCLIMATE1_5_5', 
                        'QS3_5_SCHOOLCLIMATE1_6_6', 'QS3_5_SCHOOLCLIMATE1_7_7', 
                        'QS3_5_SCHOOLCLIMATE1_8_8', 'QS3_5_SCHOOLCLIMATE1_9_9', 
                        'QS3_5_SCHOOLCLIMATE1_10_10', 'QS4_2_MEANINGFULPERSON',
                        'QS4_8_HELPSEEKING1_1_1', 'QS4_8_HELPSEEKING1_2_2', 
                        'QS4_8_HELPSEEKING1_3_3', 'QS4_8_HELPSEEKING1_4_4', 
                        'QS4_8_HELPSEEKING1_5_5', 'QS4_8_HELPSEEKING1_6_6', 
                        'QS4_8_HELPSEEKING1_7_7', 'QS4_8_HELPSEEKING1_8_8', 
                        'QS4_8_HELPSEEKING1_9_9', 'QS4_8_HELPSEEKING1_10_10', 
                        'QS4_10_MENTALWELLBE1_1_1', 'QS4_10_MENTALWELLBE1_2_2', 
                        'QS4_10_MENTALWELLBE1_3_3', 'QS4_10_MENTALWELLBE1_4_4', 
                        'QS4_10_MENTALWELLBE1_5_5', 'QS4_10_MENTALWELLBE1_6_6', 
                        'QS4_10_MENTALWELLBE1_7_7']
ordinal_chr_vars = ['QS1_22_HIGHESTEDU', 'QS2_6_MENTOREXPER', 
                    'QS2_34_SUPPORTS1_1_1', 'QS2_34_SUPPORTS1_2_2', 
                    'QS2_34_SUPPORTS1_3_3', 'QS2_34_SUPPORTS1_4_4', 
                    'QS2_34_SUPPORTS1_5_5', 'QS2_34_SUPPORTS1_6_6', 
                    'QS2_34_SUPPORTS1_7_7', 'QS2_34_SUPPORTS1_8_8', 
                    'QS2_34_SUPPORTS1_9_9', 'QS2_36_INFLUENCE1_1_1', 
                    'QS2_36_INFLUENCE1_2_2', 'QS2_36_INFLUENCE1_3_3', 
                    'QS2_36_INFLUENCE1_4_4', 'QS2_36_INFLUENCE1_5_5', 
                    'QS2_36_INFLUENCE1_6_6', 'QS2_36_INFLUENCE1_7_7', 
                    'QS2_36_INFLUENCE1_8_8', 'QS2_36_INFLUENCE1_9_9', 
                    'QS4_3_CAREERPLANNIN1_1_1', 'QS4_3_CAREERPLANNIN1_2_2', 
                    'QS4_3_CAREERPLANNIN1_3_3', 'QS4_3_CAREERPLANNIN1_4_4', 
                    'QS4_3_CAREERPLANNIN1_5_5', 'QS4_3_CAREERPLANNIN1_6_6', 
                    'QS4_3_CAREERPLANNIN1_7_7', 'QS4_3_CAREERPLANNIN1_8_8', 
                    'QS4_5_SATEDU', 'QS4_7_SOCIALCAPITAL1_1_1', 
                    'QS4_7_SOCIALCAPITAL1_2_2', 'QS4_7_SOCIALCAPITAL1_3_3', 
                    'QS4_7_SOCIALCAPITAL1_4_4', 'QS4_9_MENTALHEALTH', 
                    'QS4_11_BELONGING', 'QS4_20_MENTEEAGE', 'QS4_24_FUTUREMEN']
orinal_prefixes = ('QS1_18_', 'QS2_19_', 'QS2_20_', 'QS2_30_', 'QS2_31_',
                   'QS2_32_', 'QS2_35_', 'QS2_37_', 'QS3_1_', 'QS3_5_',
                   'QS4_2_', 'QS4_8_', 'QS4_10_', 'QS1_22_', 'QS2_6_',
                   'QS2_34_', 'QS2_36_', 'QS4_3_', 'QS4_5_', 'QS4_7_',
                   'QS4_9_', 'QS4_11_', 'QS4_20_', 'QS4_24_')

cate_vars = ['QS1_2_PROV', 'QS1_3_COMMUNITYTYPE', 'QS1_4_INDIGENOUS', 
             'QS1_5_INDIGENOUSHS', 'QS1_7_NEWCOMER', 'QS1_10_TRANSUM', 
             'QS1_11_SEXUALO', 'QS1_12_DISABIL', 'QS1_13_DISABIL', 
             'QS1_17_INCARE', 'QS1_19_HIGHSCHOOL', 'QS1_21_FURTHEDUCA', 
             'QS1_25_EMPLOYMENT', 'QS1_27_PLANNINGRE', 'QS1_28_EMPLOYMENT', 
             'QS2_1_MEANINGFULP', 'QS2_2_MEANINGFULP', 'QS2_3_PRESENCEOFM', 
             'QS2_4_MENTOR61FOR', 'QS2_5_MENTOR611PR', 'QS2_7_MENTOR611SE', 
             'QS2_8_UNMETNEED61', 'QS2_9_PRESENCEOFA', 'QS2_11_MENTOR1218', 
             'QS2_12_UNMETNEED1', 'QS2_16_FORMAT_1', 'QS2_17_TYPE_1', 
             'QS2_18_LOCATION_1', 'QS2_22_GEOLOCATI1', 'QS2_16_FORMAT_2', 
             'QS2_17_TYPE_2', 'QS2_18_LOCATION_2', 'QS2_22_GEOLOCATI2', 
             'QS2_16_FORMAT_3', 'QS2_17_TYPE_3', 'QS2_18_LOCATION_3', 
             'QS2_22_GEOLOCATI3', 'QS2_23_MOSTMEANI', 'QS2_24_MENTORAGE', 
             'QS2_28_MATCHCHOICE', 'Match_GenderIdentity', 'Match_Ethnicity', 
             'Match_CulturalBackground', 'Match_ScheduleAvailability', 
             'Match_Interests', 'Match_Goals', 'Match_Personalities', 
             'Match_LifeStruggles', 'Transition_School', 'Transition_NewSchool', 
             'Transition_NewCommunity', 'Transition_GettingDriversLicense', 
             'Transition_JobAspirations', 'Transition_GettingFirstJob', 
            'Transition_ApplyingToTradeSchool-Collge-Uni', 'Transition_IndependenceFromGuardian', 
             'Transition_FundingForTradeSchool-Collge-Uni', 'Transition_NoneOfAbove', 
             'QS2_33_TRANSITIONS1_13_13', 'QS2_33_TRANSITIONS1_14_14', 
             'QS3_4_LIFEEVENTS1_1_1', 'QS3_4_LIFEEVENTS1_2_2', 
             'QS3_4_LIFEEVENTS1_3_3', 'QS3_4_LIFEEVENTS1_4_4', 
             'QS3_4_LIFEEVENTS1_5_5', 'QS3_4_LIFEEVENTS1_6_6', 
             'QS3_4_LIFEEVENTS1_7_7', 'QS3_4_LIFEEVENTS1_8_8', 
             'QS3_4_LIFEEVENTS1_9_9', 'QS3_4_LIFEEVENTS1_10_10', 
             'QS3_4_LIFEEVENTS1_11_11', 'QS3_4_LIFEEVENTS1_12_12', 
             'QS3_4_LIFEEVENTS1_13_13', 'QS3_4_LIFEEVENTS1_14_14', 
             'QS3_4_LIFEEVENTS1_15_15', 'QS3_4_LIFEEVENTS1_16_16', 
             'QS3_4_LIFEEVENTS1_17_17', 'QS3_4_LIFEEVENTS1_18_18', 
             'QS3_4_LIFEEVENTS1_19_19', 'QS3_4_LIFEEVENTS1_20_20', 
             'QS4_1_MEANINGFULPERSON', 'QS4_4_EDUCATIONALEXPEC', 
             'QS4_6_DISAPPOINTED', 'QS4_13_LIFEEVE1_1_1', 
             'QS4_13_LIFEEVE1_2_2', 'QS4_13_LIFEEVE1_3_3', 
             'QS4_13_LIFEEVE1_4_4', 'QS4_13_LIFEEVE1_5_5', 
             'QS4_13_LIFEEVE1_6_6', 'QS4_16_FORMALVOL', 
             'QS4_17_SERVEDASM', 'QS4_18_CURRENTOR', 
             'QS4_21_MENTORING', 'QS4_22_PASTMENTO', 'QS4_25_FUTUREMEN', 
             'QS4_26_INTERNETC', 'QS4_27_INTERNETC1_1_1', 
             'QS4_27_INTERNETC1_2_2', 'QS4_27_INTERNETC1_3_3', 
             'QS4_27_INTERNETC1_4_4', 'QS4_28_INTERNETCON', 
             'QS4_29_PRIVATECONN', 'QS4_30_INTERNETCON', 
             'QS4_31_MOBILECONNE', 'QS4_32_MOBILECONNE1_1_1', 
             'QS4_32_MOBILECONNE1_2_2', 'QS4_32_MOBILECONNE1_3_3', 
             'QS4_32_MOBILECONNE1_4_4', 'QS4_33_MOBILECONNECT']

# input preparation for embedding
continuous_cols = [col for col in data.columns if col.startswith(continuous_prefixes)]
ordinal_cols = [col for col in data.columns if col.startswith(orinal_prefixes)]
ohe_cols = [col for col in data.columns if '_' in col and col not in continuous_cols + ordinal_cols]
# ohe_cols = ohe_cols.remove('QS1_28_EMPLOYMENT_calculated')

# Group OHE columns 
# (for improving interpretability--preserve semantic sturcture of each construct being tested per question)
ohe_groups = {}
for col in ohe_cols:
    if re.match(r'QS\d+_\d+_[A-Z0-9]+', col): # e.g., QS4_23_MENTEEAGE
        group = re.match(r'(QS\d+_\d+_[A-Z0-9]+)', col).group(1) 
    elif re.match(r'QS\d+_\d+_[A-Z0-9]+_\d+_\d+', col): # e.g., QS2_18_LOCATION_1_X1
        group = re.match(r'(QS_\d+_\d+_[A-Z0-9]+)', col).group(1)
    else:
        group = col.split('_')[0]
    ohe_groups.setdefault(group, []).append(col)

# Handle single-column OHE groups
single_col_groups = {group: cols for group, cols in ohe_groups.items() if len(cols) == 1}
multi_col_groups = {group: cols for group, cols in ohe_groups.items() if len(cols) > 1}
single_col_cols = [cols[0] for cols in single_col_groups.values()]
single_col_idx = [data.columns.get_loc(col) for col in single_col_cols]

# # Handle missing values and outliers
data[continuous_cols] = data[continuous_cols].fillna(data[continuous_cols].mean())
data[ordinal_cols] = data[ordinal_cols].fillna(data[ordinal_cols].mode().iloc[0])
data[ohe_cols] = data[ohe_cols].fillna(0)
data[continuous_cols] = data[continuous_cols].clip(lower=-3, upper=3)

# Let's check if all ordinal columns are scaled to [0,1] range properly!
for col in ordinal_cols:
    if data[col].min() < 0 or data[col].max() > 1:
        print(f"Rescaling {col}") 
## Verify continuous variables' preprocessing
# for col in continuous_cols:
#     assert abs(data[col].mean()) < 1e-2 and abs(data[col].std() - 1) < 1e-2, f"{col} not standardized"
# Calculate the number of columns per data type:
print(f"Continuous: {len(continuous_cols)}, Ordinal: {len(ordinal_cols)}, Categorical: {len(ohe_cols)}")


# ------------------------------------------------

# Convert to torch tensors & Split data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_X = torch.tensor(train_data.values, dtype=torch.float32)
val_X = torch.tensor(val_data.values, dtype=torch.float32)
test_X = torch.tensor(test_data.values, dtype = torch.float32)
train_dataset = TensorDataset(train_X, train_X)
val_dataset = TensorDataset(val_X, val_X)
test_dataset = TensorDataset(test_X, test_X)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# Indices
continuous_idx = [data.columns.get_loc(col) for col in continuous_cols]
ordinal_idx = [data.columns.get_loc(col) for col in ordinal_cols]
single_col_groups = {group: cols for group, cols in ohe_groups.items() if len(cols) == 1}
multi_col_groups = {group: cols for group, cols in ohe_groups.items() if len(cols) > 1}
single_col_cols = [cols[0] for cols in single_col_groups.values()]
single_col_idx = [data.columns.get_loc(col) for col in single_col_cols]

# Compute class weights for multi-column OHE groups
ohe_class_weights = {}
for group, cols in multi_col_groups.items():
    idx = [data.columns.get_loc(col) for col in cols]
    group_data = data[cols].values
    class_counts = np.sum(group_data, axis=0)
    class_counts = np.maximum(class_counts, 1)
    weights = len(data) / (len(cols) * class_counts)
    weights = np.minimum(weights, 7.0)  # Increased cap for sparse groups
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ohe_class_weights[group] = torch.tensor(weights, dtype=torch.float32).to(device)

# Define autoencoder
class MixedDataAutoencoder(nn.Module):
    def __init__(self, input_dim, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups, hidden_dims=[384, 256, 16]):
        super(MixedDataAutoencoder, self).__init__()
        self.cont_idx = continuous_idx
        self.ord_idx = ordinal_idx
        self.single_col_idx = single_col_idx
        self.multi_col_groups = multi_col_groups
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[1]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[1]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], input_dim)

        )
    
    def forward(self, x, noise_factor=0.2):  # Reduced noise for continuous
        if self.training:
            noise = torch.randn_like(x) * noise_factor
            x_noisy = x + noise
            x_noisy[:, self.cont_idx] = x_noisy[:, self.cont_idx].clamp(-3, 3)
            x_noisy[:, self.ord_idx] = x_noisy[:, self.ord_idx].clamp(0, 1)
            for cols in self.multi_col_groups.values():
                idx = [data.columns.get_loc(col) for col in cols]
                x_noisy[:, idx] = x[:, idx]
            if self.single_col_idx:
                x_noisy[:, self.single_col_idx] = x[:, self.single_col_idx]
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        outputs = {
            'continuous': decoded[:, self.cont_idx],
            'ordinal': torch.sigmoid(decoded[:, self.ord_idx]),
            'single_ohe': torch.sigmoid(decoded[:, self.single_col_idx]) if self.single_col_idx else None,
            'multi_ohe': {group: torch.softmax(decoded[:, [data.columns.get_loc(col) for col in cols]], dim=1)
                          for group, cols in self.multi_col_groups.items()}
        }
        return outputs, encoded

# Loss function
def mixed_loss(outputs, target, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups):
    mse_loss = nn.MSELoss(reduction='mean')
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    loss = 0.0
    total_units = len(continuous_idx) + len(ordinal_idx) + len(multi_col_groups) + len(single_col_idx)
    cont_weight = 1.0 if continuous_idx else 0.0  # Increased to prioritize continuous
    ord_weight = 0.2 if ordinal_idx else 0.0      # Increased further
    multi_ohe_weight = 0.7 / len(multi_col_groups) if multi_col_groups else 0.0
    single_ohe_weight = 0.5 / len(single_col_idx) if single_col_idx else 0.0
    
    if outputs['continuous'].shape[1] > 0:
        loss += mse_loss(outputs['continuous'], target[:, continuous_idx]) * cont_weight
    if outputs['ordinal'].shape[1] > 0:
        loss += mse_loss(outputs['ordinal'], target[:, ordinal_idx]) * ord_weight
    if outputs['single_ohe'] is not None:
        loss += bce_loss(outputs['single_ohe'], target[:, single_col_idx]) * single_ohe_weight
    if multi_col_groups:
        ohe_loss = 0.0
        for group, cols in multi_col_groups.items():
            idx = [data.columns.get_loc(col) for col in cols]
            ce_loss = nn.CrossEntropyLoss(weight=ohe_class_weights.get(group, None), reduction='mean')
            target_class = torch.argmax(target[:, idx], dim=1)
            ohe_loss += ce_loss(outputs['multi_ohe'][group], target_class)
        loss += ohe_loss * multi_ohe_weight
    
    return loss / total_units  # Normalize by total units

# Initialize and train
model = MixedDataAutoencoder(data.shape[1], continuous_idx, ordinal_idx, single_col_idx, multi_col_groups)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
num_epochs = 20
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = mixed_loss(outputs, x, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch[0].to(device)
            outputs, _ = model(x)
            loss = mixed_loss(outputs, x, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Evaluate (validation)
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
    outputs, encoded = model(X_tensor)
    val_loss = mixed_loss(outputs, X_tensor, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups).item()
    print(f"Final Validation Loss: {val_loss:.4f}")
    latent_features = encoded.cpu().numpy()
    print(f"Latent shape: {latent_features.shape}")
    # latent_features 
    # latent features are not directly interpretable as specific input features.
    # they are a compressed, non-linear combination of all 461 input 
    
    if continuous_idx:
        cont_pred = outputs['continuous'].cpu().numpy()
        cont_true = X_tensor[:, continuous_idx].cpu().numpy()
        print(f"Continuous MSE: {((cont_pred - cont_true) ** 2).mean():.4f}")
    if ordinal_idx:
        ord_pred = outputs['ordinal'].cpu().numpy()
        ord_true = X_tensor[:, ordinal_idx].cpu().numpy()
        print(f"Ordinal MSE: {((ord_pred - ord_true) ** 2).mean():.4f}")
    if single_col_idx:
        single_pred = outputs['single_ohe'].cpu().numpy()
        single_true = X_tensor[:, single_col_idx].cpu().numpy()
        print(f"Single-column OHE MSE: {((single_pred - single_true) ** 2).mean():.4f}")
    for group, cols in multi_col_groups.items():
        idx = [data.columns.get_loc(col) for col in cols]
        pred = outputs['multi_ohe'][group].cpu().numpy()
        true = X_tensor[:, idx].cpu().numpy()
        print(f"{group} Reconstruction Error (MSE): {((pred - true) ** 2).mean():.4f}")


# Evaluate on test set
model.eval()
total_test_loss = 0
total_test_cont_mse = 0
total_test_ord_mse = 0
total_test_single_ohe_mse = 0
test_multi_ohe_mses = {group: 0.0 for group in multi_col_groups.keys()}
print(f"Latent shape: {latent_features.shape}")
with torch.no_grad():
    for batch in test_dataloader:
        x = batch[0].to(device)
        outputs, _ = model(x)
        loss = mixed_loss(outputs, x, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups)
        print(f"Final Loss: {loss:.4f}")
        total_test_loss += loss.item()
        print(f"Total Test Loss: {total_test_loss:.4f}")
        
        if continuous_idx:
            cont_pred = outputs['continuous']
            cont_true = x[:, continuous_idx]
            total_test_cont_mse += ((cont_pred - cont_true) ** 2).mean().item()
            print(f"Test Continuous MSE: {((cont_pred - cont_true) ** 2).mean():.4f}")
        if ordinal_idx:
            ord_pred = outputs['ordinal']
            ord_true = x[:, ordinal_idx]
            total_test_ord_mse += ((ord_pred - ord_true) ** 2).mean().item()
            print(f"Test Ordinal MSE: {((ord_pred - ord_true) ** 2).mean():.4f}")
        if single_col_idx:
            single_pred = outputs['single_ohe']
            single_true = x[:, single_col_idx]
            total_test_single_ohe_mse += ((single_pred - single_true) ** 2).mean().item()
            print(f"Test Single-column OHE MSE: {((single_pred - single_true) ** 2).mean():.4f}")
        for group, cols in multi_col_groups.items():
            idx = [data.columns.get_loc(col) for col in cols]
            pred = outputs['multi_ohe'][group]
            true = x[:, idx]
            test_multi_ohe_mses[group] += ((pred - true) ** 2).mean().item()
        print(f"{group} Test Reconstruction Error (MSE): {((pred - true) ** 2).mean():.4f}")

# Final evaluation on full dataset (optional, for comparison)
with torch.no_grad():
    X_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
    outputs, encoded = model(X_tensor)
    val_loss = mixed_loss(outputs, X_tensor, continuous_idx, ordinal_idx, single_col_idx, multi_col_groups).item()
    print(f"\nFinal Validation Loss (Full Train+Val): {val_loss:.4f}")
    latent_features = encoded.cpu().numpy()
    print(f"Latent shape: {latent_features.shape}")
    
    if continuous_idx:
        cont_pred = outputs['continuous'].cpu().numpy()
        cont_true = X_tensor[:, continuous_idx].cpu().numpy()
        print(f"Continuous MSE: {((cont_pred - cont_true) ** 2).mean():.4f}")
    if ordinal_idx:
        ord_pred = outputs['ordinal'].cpu().numpy()
        ord_true = X_tensor[:, ordinal_idx].cpu().numpy()
        print(f"Ordinal MSE: {((ord_pred - ord_true) ** 2).mean():.4f}")
    if single_col_idx:
        single_pred = outputs['single_ohe'].cpu().numpy()
        single_true = X_tensor[:, single_col_idx].cpu().numpy()
        print(f"Single-column OHE MSE: {((single_pred - single_true) ** 2).mean():.4f}")
    for group, cols in multi_col_groups.items():
        idx = [data.columns.get_loc(col) for col in cols]
        pred = outputs['multi_ohe'][group].cpu().numpy()
        true = X_tensor[:, idx].cpu().numpy()
        print(f"{group} Reconstruction Error (MSE): {((pred - true) ** 2).mean():.4f}")

# Examine Silhouette score to explore clustering quality! 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

results = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(latent_features)
    score = silhouette_score(latent_features, clusters)
    results.append((k, score))
    print(f"k={k}, Silhouette Score: {score:.4f}") # NOT GOOD!
best_k = max(results, key=lambda x: x[1])[0] # get the best K from tuning
print(f"Best k: {best_k}")

# PCA for further compression:
pca = PCA(n_components = 2) 
reduced_features = pca.fit_transform(latent_features)
kmeans = KMeans(n_clusters = best_k, random_state=42) #best_k
clusters = kmeans.fit_predict(reduced_features)
print(f"Silhouette Score (PCA, n_components = 2 ): {silhouette_score(reduced_features, clusters):.4f}")
# Not idea score still--typically, >0.5 indicates good clustering

# Visualize PCA clusters
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', s=50)
plt.title('Cluster Visualization (PCA, k=2, latent_dim=16)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
# plt.savefig('clusters_pca_k2.png')
plt.show()
plt.clf()
# PCA loading
print(pd.DataFrame(pca.components_, columns=[f'latent_{i}' for i in range(16)], index=['PC1', 'PC2']))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_features = tsne.fit_transform(latent_features)
clusters_tsne = kmeans.fit_predict(tsne_features)

plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=clusters_tsne, cmap='viridis', s=50)
plt.title('Cluster Visualization (t-SNE, k=2, latent_dim=16)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.savefig('clusters_tsne_k2.png')
plt.show()
plt.clf()



# latent_features is (1915, 16) from the autoencoder
latent_df = pd.DataFrame(latent_features, columns=[f'latent_{i}' for i in range(16)])
data_with_latent = pd.concat([data.reset_index(drop=True), latent_df], axis=1)
correlations = data_with_latent.corr()[latent_df.columns].loc[data.columns]  # Shape: (461, 16)

# Find top input features for each latent dimension
for latent_col in latent_df.columns:
    top_features = correlations[latent_col].abs().sort_values(ascending=False).head(5)
    print(f"\nTop 5 input features correlated with {latent_col}:")
    print(top_features)

# Find top input features overall (highest correlations across all latent dimensions)
max_correlations = correlations.abs().max().sort_values(ascending=False).head(10)
print("\nTop 10 input features with highest correlations to any latent dimension:")
print(max_correlations) 
#Strong Correlations! good for interpretability but likely indicate overfitting to certain variable...

# Analyze clusters to check for meaningful groupings (e.g., by mentorship).
data['cluster'] = clusters
key_features = ['QS4_4_EDUCATIONALEXPEC_X7', 'Transition_IndependenceFromGuardian_Yes', 'QS2_32_MENTORINGENG1_14_14']
for cluster in range(2):
    cluster_data = data[data['cluster'] == cluster]
    print(f"Cluster {cluster} (n={len(cluster_data)}):")
    print(cluster_data[key_features].mean().sort_values(ascending=False))
    #This suggests both clusters have high mentorship engagement, with Cluster 0 slightly higher.
    #The small differences in means indicate the clusters (k=2, silhouette score: ~0.39 with PCA, n_components=2) are not distinctly separated
    #The low values for QS4_4_EDUCATIONALEXPEC_X7 suggest it may have high reconstruction error?
    #Sparse Data: Survey data with many OHE variables (e.g., Transition_IndependenceFromGuardian_Yes) is sparse, reducing inter-sample differences and cluster distinctness?
    #Limited Cluster Structure: The data may naturally form only broad groups (k=2)

    #The lack of distinct separation suggests the latent features don’t capture strong, unique patterns for clustering:(

# ---PCA loading extraction:
# correlations = data_with_latent.corr()[latent_df.columns].loc[data.columns]
# for pc in ['PC1', 'PC2']:
#     print(f"\nTop Input Features for {pc}:")
#     pc_loadings = loadings.loc[pc].abs()
#     top_latent_dims = pc_loadings.sort_values(ascending=False).head(3).index
#     for dim in top_latent_dims:
#         print(f"\n{dim}:")
#         print(correlations[dim].abs().sort_values(ascending=False).head(5))