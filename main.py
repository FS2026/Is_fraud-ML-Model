import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, classification_report, precision_recall_curve)
 
# Hyperparameters
TRANSACTIONS_FILE = 'transactions.json'
MERCHANTS_FILE = 'merchants.csv'
USERS_FILE = 'users.csv'
MODEL_SAVE_PATH = 'optimized_fraud_detection_model.pth'
NUM_EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 0.0015
WEIGHT_DECAY = 1e-5
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
LAYER_1_SIZE = 256
LAYER_2_SIZE = 128
LAYER_3_SIZE = 64
DROPOUT_1 = 0.5
DROPOUT_2 = 0.4
DROPOUT_3 = 0.3
 
# Data loading
transactions = pd.read_json(TRANSACTIONS_FILE, lines=True)
merchants = pd.read_csv(MERCHANTS_FILE)
users = pd.read_csv(USERS_FILE, parse_dates=['signup_date'])
 
merchants = merchants.rename(columns={'country': 'country_merchant', 'category': 'category_merchant', 'has_fraud_history': 'has_fraud_history_merchant'})
users = users.rename(columns={'country': 'country_user'})
 
# Data merging
df = (
    transactions
    .assign(
        latitude=transactions.location.map(lambda d: d.get('lat') ),
        longitude=transactions.location.map(lambda d: d.get('long'))
    )
    .drop(columns='location', errors='ignore')
    .merge(users, on='user_id', how='left')
    .merge(merchants, on='merchant_id', how='left')
)
print("1. Data loaded")
 
# Features engeenirng time
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['is_fraud'] = df['is_fraud'].astype(int)
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)
 
df['transaction_hour'] = df['timestamp'].dt.hour
df['transaction_day_of_week'] = df['timestamp'].dt.dayofweek
df['transaction_day_of_month'] = df['timestamp'].dt.day
df['transaction_month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['transaction_day_of_week'] >= 5).astype(int)
df['is_night_transaction'] = ((df['transaction_hour'] < 6) | (df['transaction_hour'] > 22)).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour']/24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour']/24.0)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['transaction_day_of_week']/7.0)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['transaction_day_of_week']/7.0)
df['dayofmonth_sin'] = np.sin(2 * np.pi * df['transaction_day_of_month']/df['timestamp'].dt.days_in_month)
df['dayofmonth_cos'] = np.cos(2 * np.pi * df['transaction_day_of_month']/df['timestamp'].dt.days_in_month)
df['month_sin'] = np.sin(2 * np.pi * df['transaction_month']/12.0)
df['month_cos'] = np.cos(2 * np.pi * df['transaction_month']/12.0)
 
df['user_account_age_days'] = (df['timestamp'] - df['signup_date']).dt.days
 
# Srtring to frequency
freq_cols = ['user_id', 'merchant_id', 'category_merchant', 'device', 'payment_method', 'country_merchant', 'country_user']
for col in freq_cols:
    if col in df.columns:
        col_filled = df[col].astype(str)
        counts = col_filled.value_counts(normalize=True)
        df[f'{col}_freq_global'] = col_filled.map(counts)
 
# Features engeenirng user
df['user_count_hist'] = df.groupby('user_id').cumcount()
df['user_fraud_count_hist'] = df.groupby('user_id')['is_fraud'].transform(lambda x: x.shift(1).expanding().sum())
df['user_fraud_rate_hist'] = (df['user_fraud_count_hist'] / (df['user_count_hist'].replace(0, 1)))
df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
df['user_avg_amount_hist'] = df.groupby('user_id')['amount'].transform(lambda x: x.shift(1).expanding().mean())
df['user_median_amount_hist'] = df.groupby('user_id')['amount'].transform(lambda x: x.shift(1).expanding().median())
df['amount_vs_user_median_hist_ratio'] = (df['amount'] / (df['user_median_amount_hist'].replace(0, 0.01)))
df['amount_deviation_from_user_median'] = df['amount'] - df['user_median_amount_hist']
 
# Features engeenirng merchant
df['merchant_count_hist'] = df.groupby('merchant_id').cumcount()
df['merchant_fraud_count_hist'] = df.groupby('merchant_id')['is_fraud'].transform(lambda x: x.shift(1).expanding().sum())
df['merchant_fraud_rate_hist'] = (df['merchant_fraud_count_hist'] / (df['merchant_count_hist'].replace(0,1)))
df['time_since_last_merchant_transaction'] = df.groupby('merchant_id')['timestamp'].diff().dt.total_seconds()
df['merchant_avg_amount_hist'] = df.groupby('merchant_id')['amount'].transform(lambda x: x.shift(1).expanding().mean())
df['amount_vs_merchant_avg_hist_ratio'] = (df['amount'] / (df['merchant_avg_amount_hist'].replace(0, 0.01)))
df['user_merchant_count_hist'] = df.groupby(['user_id', 'merchant_id']).cumcount()
df['time_since_last_user_merchant'] = df.groupby(['user_id', 'merchant_id'])['timestamp'].diff().dt.total_seconds()
df['user_country_merchant_country'] = (df['country_user'] == df['country_merchant']).astype(int)
df['amount_vs_merchant_reported_avg_ratio'] = (df['amount'] / (df['avg_transaction_amount'].replace(0, 0.01)))
 
# Excluding unnecessary colums
target = df.pop('is_fraud')
cols_to_exclude = [
    'transaction_id', 'user_id', 'merchant_id', 'timestamp', 'signup_date',
    'latitude', 'longitude', 'transaction_hour', 'transaction_day_of_week',
    'transaction_day_of_month', 'transaction_month', 'currency',
]
cols_to_exclude = [col for col in cols_to_exclude if col in df.columns]
 
categorical = [
    'channel', 'device', 'payment_method', 'is_international',
    'is_first_time_merchant', 'sex', 'education', 'primary_source_of_income',
    'country_user', 'category_merchant', 'country_merchant', 'has_fraud_history_merchant'
]
 
# Final df
final_categorical_feature_names = [col for col in categorical if col in df.columns and col not in cols_to_exclude]
final_categorical_feature_names = sorted(list(set(final_categorical_feature_names)))
 
final_numerical_feature_names = [
    col for col in df.columns
    if col not in cols_to_exclude and
        col not in final_categorical_feature_names and
        pd.api.types.is_numeric_dtype(df[col])
]
final_numerical_feature_names = sorted(list(set(final_numerical_feature_names)))
 
# Features and label
all_feature_names_final = final_numerical_feature_names + final_categorical_feature_names
X = df[all_feature_names_final].copy()
y = target.copy()
 
# Data encoding
for col in final_categorical_feature_names:
    X[col] = X[col].astype(str)
    X[col] = LabelEncoder().fit_transform(X[col])
    X[col] = X[col].astype(float)
print("2. Data Encoded")
 
# Scaling and completing data
full_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_processed_array = full_transformer.fit_transform(X)
 
# Converting to numpy
y_numpy = y.values
 
# Dividing data into sets
X_val, X_test, y_val, y_test = train_test_split(
    X_processed_array, y_numpy, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_numpy
)
val_split_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_val, y_val, test_size=val_split_ratio,
    random_state=RANDOM_STATE, stratify=y_val
)
 
# Random seed and selecting device
torch.manual_seed(RANDOM_STATE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("3. Selected {device}")
 
# Converting and loading data to pytorch
train_ds = TensorDataset(torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).float().to(device))
val_ds   = TensorDataset(torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).float().to(device))
test_ds  = TensorDataset(torch.from_numpy(X_test).float().to(device), torch.from_numpy(y_test).float().to(device))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print("4. Data moved to pytorch")
 
# Neural Network
class Fraud(nn.Module):
    def __init__(self, in_features, l1=LAYER_1_SIZE, l2=LAYER_2_SIZE, l3=LAYER_3_SIZE, d1=DROPOUT_1, d2=DROPOUT_2, d3=DROPOUT_3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, l1),
            nn.BatchNorm1d(l1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(l1, l2),
            nn.BatchNorm1d(l2), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(l2, l3),
            nn.BatchNorm1d(l3), nn.ReLU(), nn.Dropout(d3),
            nn.Linear(l3, 1)
        )
    def forward(self, x):
        return self.net(x)
 
model = Fraud(X_train.shape[1]).to(device)
 
# Number of frauds
num_pos_samples = np.sum(y_train == 1)
num_neg_samples = np.sum(y_train == 0)
pos_weight_val = num_neg_samples / num_pos_samples
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
 
# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
 
# Variables inicialisation
best_val_f1 = 0
epochs_no_improve = 0
best_model_state = None
 
# Trainig loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
 
# Model evaluation
    model.eval()
    val_preds_for_f1, val_probs, val_truths = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            out = model(x_batch)
            probs_batch = torch.sigmoid(out)
            preds_batch_for_f1 = (probs_batch > 0.5).int()
            val_preds_for_f1.extend(preds_batch_for_f1.cpu().numpy().flatten())
            val_probs.extend(probs_batch.cpu().numpy().flatten())
            val_truths.extend(y_batch.cpu().numpy().flatten())
    # Calculating F1 score
    current_val_f1 = f1_score(val_truths, val_preds_for_f1, zero_division=0)
    scheduler.step(current_val_f1)
    # Printing current parameters
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f} - F1: {current_val_f1:.4f} - LR: {optimizer.param_groups[0]['lr']:.1e}")
 
    if current_val_f1 > best_val_f1:
        best_val_f1 = current_val_f1
        best_model_state = model.state_dict()
 
# Saving model
model.load_state_dict(best_model_state)
torch.save(best_model_state, MODEL_SAVE_PATH)
print(f"Best model saved: {MODEL_SAVE_PATH} (Val F1: {best_val_f1:.4f})")
 
# Model evaluation threshold
model.eval()
val_probs, val_truths = [], []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        probs_batch = torch.sigmoid(out)
        val_probs.extend(probs_batch.cpu().numpy().flatten())
        val_truths.extend(y_batch.cpu().numpy().flatten())
 
 
precisions, recalls, thresholds = precision_recall_curve(val_truths, val_probs)
if len(thresholds) == len(precisions):
    thresholds = thresholds[:-1]
precisions = precisions[:len(thresholds)]
recalls = recalls[:len(thresholds)]
 
# Finding best optimal threshold
f1_scores_thresh = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
optimal_threshold = 0.5
if len(f1_scores_thresh) > 0:
    optimal_idx = np.argmax(f1_scores_thresh)
    if optimal_idx < len(thresholds):
          optimal_threshold = thresholds[optimal_idx]
    else:
        if len(thresholds)>0: optimal_threshold = thresholds[-1]
 
 
# Test set evaluation
test_probs, test_truths = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        probs_batch = torch.sigmoid(out)
        test_probs.extend(probs_batch.cpu().numpy().flatten())
        test_truths.extend(yb.cpu().numpy().flatten())
 
# Evaluation Result calculation
test_preds_optimal_thresh = (np.array(test_probs) > optimal_threshold).astype(int)
test_truths_np = np.array(test_truths)
test_probs_np = np.array(test_probs)
 
cm = confusion_matrix(test_truths_np, test_preds_optimal_thresh, labels=[0, 1])
tn, fp, fn, tp = 0,0,0,0
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    if len(np.unique(test_truths_np)) == 1:
        if np.unique(test_truths_np)[0] == 0:
            tn = cm[0,0]
            fp = cm[0,1] if cm.shape[1] > 1 else 0
        else:
            fn = cm[0,0] if cm.shape[0] > 1 else 0
            tp = cm[-1,-1]
 
accuracy = accuracy_score(test_truths_np, test_preds_optimal_thresh)
precision = precision_score(test_truths_np, test_preds_optimal_thresh, zero_division=0)
recall = recall_score(test_truths_np, test_preds_optimal_thresh, zero_division=0)
f1_final = f1_score(test_truths_np, test_preds_optimal_thresh, zero_division=0)
roc_auc = roc_auc_score(test_truths_np, test_probs_np)
 
print(f"Optimal threshold = {optimal_threshold:.4f}")
print(f"(TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1_final:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(test_truths_np, test_preds_optimal_thresh, target_names=['Not Fraud', 'Fraud'], zero_division=0))