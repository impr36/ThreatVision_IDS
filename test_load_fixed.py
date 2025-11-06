# test_load_fixed.py
from utils import load_nsl_kdd, map_labels
train_df = load_nsl_kdd('./KDDTrain+.txt')
print(f"After load: Sample label values: {train_df['label'].head(3).tolist()}")
train_df = map_labels(train_df)
print(f"After mapping: shape {train_df.shape}, unique cat_labels: {sorted(train_df['cat_label'].unique())}")
# Expected: ~125k rows, ['DoS', 'Probe', 'R2L', 'U2R', 'normal']