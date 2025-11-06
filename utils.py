# utils.py (corrected top section)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# 41 features for NSL-KDD (added 'duration' at start)
FEATURE_NAMES = [
    'duration', 
    'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
# 41 features + label = 42 columns
ALL_COLS = FEATURE_NAMES + ['label', 'difficulty']

# Expanded attack mapping 
ATTACK_TO_CATEGORY = {
    # DoS
    "back":"DoS", "land":"DoS", "neptune":"DoS", "pod":"DoS", "smurf":"DoS", "teardrop":"DoS",
    "apache2":"DoS", "udpstorm":"DoS", "processtable":"DoS", "mailbomb":"DoS",
    # Probe
    "satan":"Probe", "ipsweep":"Probe", "nmap":"Probe", "portsweep":"Probe", "mscan":"Probe", "saint":"Probe",
    "snmpgetattack":"Probe", "named":"Probe",
    # R2L
    "guess_passwd":"R2L", "ftp_write":"R2L", "imap":"R2L", "phf":"R2L", "multihop":"R2L",
    "warezmaster":"R2L", "warezclient":"R2L", "spy":"R2L", "xlock":"R2L", "xsnoop":"R2L",
    "snmpguess":"R2L", "sendmail":"R2L", "xterm":"R2L", "ps":"R2L", "sqlattack":"R2L",
    # U2R
    "buffer_overflow":"U2R", "loadmodule":"U2R", "rootkit":"U2R", "perl":"U2R", "httptunnel":"U2R"
}

def attack_to_category(label):
    if label == "normal" or label == "normal.":
        return "normal"
    l = str(label).strip().strip(".")
    return ATTACK_TO_CATEGORY.get(l, "unknown")

def load_nsl_kdd(filepath, names=ALL_COLS):
    df = pd.read_csv(filepath, names=names, header=None, sep=",", low_memory=False)
    df = df.dropna().reset_index(drop=True)
    # Drop the difficulty column (last one, numeric 0-21)
    df = df.drop('difficulty', axis=1)
    df['label'] = df['label'].astype(str).str.strip()
    print(f"✅ Loaded {filepath} successfully with shape {df.shape}") 
    print(f"Sample labels: {df['label'].unique()[:5]}")  
    return df

def map_labels(df):
    df['cat_label'] = df['label'].apply(lambda x: attack_to_category(x))
    # drop unknown if you want; we'll keep unknown as its own class or drop
    df = df[df['cat_label'] != 'unknown']
    return df

def preprocess(df, fit_scaler=False, scaler_path=None, enc_path=None):

    # Separate features and labels
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in FEATURE_NAMES if c not in cat_cols]

    # 🧹 Clean categorical data (remove extra spaces and NaNs)
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
        df = df[df[c] != '']  # remove empty rows

    # Verify we have data left
    if df.shape[0] == 0:
        raise ValueError("No samples left after cleaning categorical columns — check file format.")

    X = df[FEATURE_NAMES].copy()
    y = df['cat_label'].copy()

    # One-hot encode categorical columns
    if fit_scaler or not os.path.exists(enc_path or "ohe.joblib"):
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(X[cat_cols])
        joblib.dump(ohe, enc_path or "ohe.joblib")
    else:
        ohe = joblib.load(enc_path or "ohe.joblib")

    X_cat = ohe.transform(X[cat_cols])

    # Scale numerical columns
    if fit_scaler or not os.path.exists(scaler_path or "scaler.joblib"):
        scaler = StandardScaler()
        scaler.fit(X[num_cols])
        joblib.dump(scaler, scaler_path or "scaler.joblib")
    else:
        scaler = joblib.load(scaler_path or "scaler.joblib")

    X_num = scaler.transform(X[num_cols])

    # Merge numerical + categorical arrays
    X_proc = np.hstack([X_num, X_cat])

    print(f"Processed features shape: {X_proc.shape}")  # Debug: Expect ~ (n, 122)

    return X_proc, y.values, scaler, ohe

def encode_labels(y, label_encoder_path=None, fit=False):
    le = None
    if label_encoder_path:
        try:
            le = joblib.load(label_encoder_path)
        except:
            le = None
    if fit or le is None:
        le = LabelEncoder()
        le.fit(y)
        if label_encoder_path:
            joblib.dump(le, label_encoder_path)
    y_enc = le.transform(y)
    return y_enc, le

def pad_features_to_shape(X, target_size=122):  # Full post-onehot size
    if X.shape[1] >= target_size:
        return X[:, :target_size]
    pad = np.zeros((X.shape[0], target_size - X.shape[1]))
    return np.hstack([X, pad])

def create_sequences(X, y, seq_len=10, stride=1, label_mode='majority', apply_smote=False):
    if apply_smote:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"SMOTE: Oversampled from {Counter(y)} to {Counter(y_res)}")
        X, y = X_res, y_res  # Use resampled for sequences
    
    n_samples, n_features = X.shape
    seqs = []
    labs = []
    for start in range(0, n_samples - seq_len + 1, stride):
        end = start + seq_len
        seq = X[start:end]
        lab_seg = y[start:end]
        if label_mode == 'last':
            lab = lab_seg[-1]
        else:
            lab = Counter(lab_seg).most_common(1)[0][0]
        seqs.append(seq)
        labs.append(lab)
    return np.array(seqs), np.array(labs)
