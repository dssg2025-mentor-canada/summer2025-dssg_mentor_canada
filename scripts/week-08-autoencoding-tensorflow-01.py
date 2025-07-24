import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# import tensorflow as tf
# import tensorflow.keras.layers as layers
# import tensorflow.keras.models as models


# import data
data = pd.read_csv('../../dssg-2025-mentor-canada/Data/faiss_knn_imputed_normalize_train_dataset.csv')

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

# Convert to torch tensors
X = torch.tensor(data.values, dtype=torch.float32)
dataset = TensorDataset(X, X)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Indices for loss calculation
continuous_idx = [data.columns.get_loc(col) for col in continuous_cols]
ordinal_idx = [data.columns.get_loc(col) for col in ordinal_cols]

# Define autoencoder
class MixedDataAutoencoder(nn.Module):
    def __init__(self, input_dim, continuous_idx, ordinal_idx, ohe_groups, hidden_dims=[512, 384, 256]):
        super(MixedDataAutoencoder, self).__init__()
        self.cont_idx = continuous_idx
        self.ord_idx = ordinal_idx
        self.ohe_groups = ohe_groups
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        outputs = {
            'continuous': decoded[:, self.cont_idx],
            'ordinal': torch.sigmoid(decoded[:, self.ord_idx]),
            'ohe': {group: torch.softmax(decoded[:, [data.columns.get_loc(col) for col in cols]], dim=1)
                    for group, cols in self.ohe_groups.items()}
        }
        return outputs, encoded
        
# Loss function
def mixed_loss(outputs, target):
    mse_loss = nn.MSELoss(reduction = 'mean')
    ce_loss = nn.CrossEntropyLoss(reduction = 'mean')
    loss = 0.0
    if outputs['continuous'].shape[1] > 0:
        loss += mse_loss(outputs['continuous'], target[:, continuous_idx])
    if outputs['ordinal'].shape[1] > 0:
        loss += mse_loss(outputs['ordinal'], target[:, ordinal_idx])
    for group, cols in ohe_groups.items():
        idx = [data.columns.get_loc(col) for col in cols]
        loss += ce_loss(outputs['ohe'][group], target[:, idx])
    return loss

# Initialize and train
model = MixedDataAutoencoder(data.shape[1], continuous_idx, ordinal_idx, ohe_groups)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        outputs, _ = model(x)
        loss = mixed_loss(outputs, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    X_tensor = X.to(device)
    outputs, encoded = model(X_tensor)
    val_loss = mixed_loss(outputs, X_tensor).item()
    print(f"Validation Loss: {val_loss:.4f}")
    latent_features = encoded.cpu().numpy()
    print(f"Latent shape: {latent_features.shape}")