# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import (load_nsl_kdd, map_labels, preprocess, encode_labels,
                   create_sequences, pad_features_to_shape)
from model import build_cnn_lstm
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter

def plot_conf_mat(y_true, y_pred, labels, out_file="conf_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

def main(args):
    # Load
    print("Loading datasets...")
    train_df = load_nsl_kdd(args.train)
    test_df = load_nsl_kdd(args.test)

    # Map labels to categories and optionally drop unknown
    train_df = map_labels(train_df)
    test_df = map_labels(test_df)

    # Optionally shuffle (important before creating sequences)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Preprocess (fit on train)
    print("Preprocessing...")
    X_train_raw, y_train_raw, scaler, ohe = preprocess(train_df, fit_scaler=True, scaler_path=args.scaler, enc_path=args.ohe)
    X_test_raw, y_test_raw, _, _ = preprocess(test_df, fit_scaler=False, scaler_path=args.scaler, enc_path=args.ohe)

    # Label encoding
    y_train_enc, le = encode_labels(y_train_raw, label_encoder_path=args.label_enc, fit=True)
    y_test_enc, _ = encode_labels(y_test_raw, label_encoder_path=args.label_enc, fit=False)

    # Save label classes
    classes = list(le.classes_)
    joblib.dump(classes, args.classes_out)

    # Pad to full features (no 2D)
    X_train_raw = pad_features_to_shape(X_train_raw, target_size=args.feature_area)  # 122
    X_test_raw = pad_features_to_shape(X_test_raw, target_size=args.feature_area)

    # 80/20 train/val split from train (synopsis) - FIXED: Proper 80/20 without overwriting
    X_temp, X_val_raw, y_temp, y_val_enc = train_test_split(X_train_raw, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc)
    X_train_raw = X_temp  # 80% for train
    y_train_enc = y_temp

    # Create sequences with SMOTE on train
    seq_len = args.seq_len
    print(f"Creating sequences with seq_len={seq_len} and SMOTE on train...")
    X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_enc, seq_len=seq_len, stride=1, label_mode='majority', apply_smote=True)
    X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_enc, seq_len=seq_len, stride=1, label_mode='majority', apply_smote=False)
    X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_enc, seq_len=seq_len, stride=1, label_mode='majority', apply_smote=False)

    # No 2D reshape—direct 1D input: (n_seq, seq_len, n_feat)
    n_features = X_train_seq.shape[2]
    print(f"1D Input shape: (samples, {seq_len}, {n_features})")

    n_classes = len(classes)
    y_train_cat = to_categorical(y_train_seq, num_classes=n_classes)
    y_val_cat = to_categorical(y_val_seq, num_classes=n_classes)
    y_test_cat = to_categorical(y_test_seq, num_classes=n_classes)

    print("Building model...")
    model = build_cnn_lstm(input_shape=(seq_len, n_features), n_classes=n_classes,
                           lstm_units=args.lstm_units, dropout_rate=args.dropout, 
                           bidirectional=args.bidirectional, l2_reg=0.001)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Lower initial LR
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_seq), y=y_train_seq)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Callbacks (increased patience for better convergence)
    ckpt = ModelCheckpoint(args.model_out, monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)

    print("Training with class weights...")
    history = model.fit(X_train_seq, y_train_cat,
                        validation_data=(X_val_seq, y_val_cat),  # Use val, not test
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        class_weight=class_weight_dict,  # Penalize majority errors
                        callbacks=[ckpt, es, rlr],
                        verbose=1)

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred_prob = model.predict(X_test_seq)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test_seq, y_pred)
    print("Test accuracy:", acc)
    print(classification_report(y_test_seq, y_pred, target_names=classes, digits=4))

    plot_conf_mat(y_test_seq, y_pred, labels=classes, out_file="confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")
    # Save final model if not already saved
    model.save(args.model_final)
    print("Model saved to", args.model_final)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='KDDTrain+.txt path')
    parser.add_argument('--test', required=True, help='KDDTest+.txt path')
    parser.add_argument('--model_out', default='best_model.h5')
    parser.add_argument('--model_final', default='final_model.h5')
    parser.add_argument('--scaler', default='scaler.joblib')
    parser.add_argument('--ohe', default='ohe.joblib')
    parser.add_argument('--label_enc', default='label_encoder.joblib')
    parser.add_argument('--classes_out', default='classes.joblib')
    parser.add_argument('--seq_len', type=int, default=20)  # Longer for better temporals
    parser.add_argument('--feature_area', type=int, default=122)  # Full post-onehot
    parser.add_argument('--lstm_units', type=int, default=256)  # Larger for capacity
    parser.add_argument('--dropout', type=float, default=0.6)  # Higher to reduce overfit
    parser.add_argument('--epochs', type=int, default=100)  # More epochs
    parser.add_argument('--batch_size', type=int, default=32)  # Smaller for stability
    parser.add_argument('--bidirectional', action='store_true', default=True)  # Enable by default
    args = parser.parse_args()
    main(args)