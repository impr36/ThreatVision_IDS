# inference.py
import numpy as np
import joblib
from utils import pad_features_to_shape, reshape_for_cnn_time_distributed
from model import build_cnn_lstm
import tensorflow as tf

# Example usage:
# python inference.py --model final_model.h5 --classes classes.joblib --scaler scaler.joblib --ohe ohe.joblib --label_enc label_encoder.joblib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--classes', default='classes.joblib')
parser.add_argument('--scaler', default='scaler.joblib')
parser.add_argument('--ohe', default='ohe.joblib')
parser.add_argument('--label_enc', default='label_encoder.joblib')
parser.add_argument('--seq_len', type=int, default=10)
parser.add_argument('--feature_h', type=int, default=7)
parser.add_argument('--feature_w', type=int, default=6)
args = parser.parse_args()

classes = joblib.load(args.classes)
scaler = joblib.load(args.scaler)
ohe = joblib.load(args.ohe)
le = joblib.load(args.label_enc)

# Replace this with real sample(s) — here we craft a fake batch from random numbers
# In practice, build the sequence with the same preprocessing pipeline as training.
seq_len = args.seq_len
n_feat = args.feature_h * args.feature_w

# dummy sequence of zeros (trusted benign)
sample_seq = np.zeros((1, seq_len, n_feat))
# reshape for model
X_td = reshape_for_cnn_time_distributed(sample_seq, seq_len=seq_len, feature_2d=(args.feature_h, args.feature_w))
model = tf.keras.models.load_model(args.model)
pred = model.predict(X_td)
pred_label = classes[np.argmax(pred)]
print("Predicted class:", pred_label)
