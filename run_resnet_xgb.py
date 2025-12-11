# run_resnet_xgb.py
"""
ResNet50 (transfer learning) + Dense(128) head + XGBoost hybrid pipeline.
- Stratified 70/30 split (copies files)  <-- now executed ONLY when script run directly
- Augmentation (strong)
- Epoch experiments: [5,10,20,50]
- Save models, feature models, xgboost models, confusion matrices, results CSV
- Single-image inference returning class percentages (import-safe)
"""
import os, random, shutil, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import joblib

# ========== USER CONFIG - EDIT ONLY THIS ==========
DATA_ROOT = r"C:\\Users\AVICHAL TRIVEDI\\Documents\\Citrus Disease Classification using ResNet50 + XGBoost\\Orange Dataset\\dataset\\trainandtest"  # <--- set this
WORK_DIR = "./final_runs_resnet"
SPLIT_DIR = os.path.join(WORK_DIR, "data_split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16           # use 32 if you have GPU & lots of RAM
SEED = 42
TEST_SIZE = 0.30          # 70/30 split
EPOCHS_LIST = [5, 10, 20, 50]
HEAD_FREEZE_EPOCHS = 10   # train head for up to 10 epochs, then fine-tune remainder
XGB_PARAMS = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, "use_label_encoder": False, "eval_metric": "mlogloss"}
# ===================================================

# reproducibility
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# harmless top-level setup (creates output dir only; no heavy I/O)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# ---------- Augmentation (defined at top-level; does not trigger dataset load) ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.12),
], name="data_augmentation")

# ---------- ResNet preprocess import (needed by prediction function) ----------
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ---------- model builder (ResNet50 base + Dense(128) head) ----------
def build_resnet_head(num_classes, dropout_rate=0.5, lr_head=1e-4):
    base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling='avg')
    base.trainable = False  # freeze base initially
    inp = base.input
    x = base.output
    x = layers.Dense(128, activation='relu', name='feat_dense')(x)
    x = layers.Dropout(dropout_rate, name='feat_dropout')(x)
    out = layers.Dense(num_classes, activation='softmax', name='softmax_out')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_head),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base

# ---------- feature extraction helper ----------
def extract_features_from_model(feature_model, dataset):
    X_list = []
    y_list = []
    for imgs, labels in dataset:
        feats = feature_model.predict(imgs, verbose=0)
        X_list.append(feats)
        y_list.append(np.argmax(labels.numpy(), axis=1))
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

# ---------- Lightweight helper to get class names (used by predict function) ----------
def _get_classes_from_data_root():
    # Returns sorted class folder names. This is fast and safe (no heavy file operations).
    p = Path(DATA_ROOT)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    return classes

# ---------- Single-image inference helper (import-safe) ----------
from tensorflow.keras.preprocessing import image as keras_image

def predict_single_image(image_path, epochs_to_use=50):
    """
    Load saved feature model and xgboost model for the given epoch run and predict percentages.
    epochs_to_use must match one of the runs (5,10,20,50).
    This function is safe to import and does NOT trigger training/splitting.
    """
    prefix = f"resnet_ep{epochs_to_use}"
    feat_path = os.path.join(WORK_DIR, f"featmod_{prefix}.h5")
    xgb_path = os.path.join(WORK_DIR, f"xgb_{prefix}.joblib")
    if not os.path.exists(feat_path) or not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Models for epoch run {epochs_to_use} not found. Run the training script first.")

    # load models (fast)
    feat_mod = tf.keras.models.load_model(feat_path)
    xgb_model = joblib.load(xgb_path)

    # preprocess image
    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0).astype('float32')
    arr = resnet_preprocess(arr)

    feats = feat_mod.predict(arr, verbose=0)
    probs = xgb_model.predict_proba(feats)[0]

    classes = _get_classes_from_data_root()
    return {cls: float(round(p * 100.0, 3)) for cls, p in zip(classes, probs)}

# ======================================================================
# ========================= EVERYTHING BELOW IS MAIN ====================
# ======================================================================
# All heavy processing (discover files, split, create datasets, training)
# is executed ONLY when the user runs this file directly.
if __name__ == "__main__":

    # ---------- Discover classes & files ----------
    classes = sorted([d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
    print("Classes:", classes)
    all_files = []
    all_labels = []
    for c in classes:
        for f in (Path(DATA_ROOT)/c).glob("*"):
            if f.is_file():
                all_files.append(str(f))
                all_labels.append(c)
    print(f"Total images found: {len(all_files)}")

    # ---------- Stratified split (70/30) ----------
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=TEST_SIZE, random_state=SEED, stratify=all_labels
    )

    def prepare_split(root_out, files, labels):
        p = Path(root_out)
        if p.exists():
            shutil.rmtree(p)
        for fp, lbl in zip(files, labels):
            out_dir = p / lbl
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / Path(fp).name
            if not dst.exists():
                shutil.copy(fp, dst)

    print("Creating split folders (copying files)...")
    prepare_split(Path(SPLIT_DIR)/"train", train_files, train_labels)
    prepare_split(Path(SPLIT_DIR)/"test", test_files, test_labels)

    print("Train counts:", {c: len(list((Path(SPLIT_DIR)/'train'/c).glob('*'))) for c in classes})
    print("Test counts: ", {c: len(list((Path(SPLIT_DIR)/'test'/c).glob('*'))) for c in classes})

    # ---------- Datasets ----------
    AUTOTUNE = tf.data.AUTOTUNE

    def make_datasets(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size, shuffle=True, seed=SEED
        )
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir, label_mode='categorical', image_size=img_size, batch_size=batch_size, shuffle=False
        )
        return train_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE)

    train_dir = Path(SPLIT_DIR)/"train"
    test_dir = Path(SPLIT_DIR)/"test"
    train_ds_raw, test_ds_raw = make_datasets(str(train_dir), str(test_dir))

    def prepare_for_training(dataset, augment=False):
        ds = dataset.map(lambda x,y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x,y: (resnet_preprocess(x), y), num_parallel_calls=AUTOTUNE)
        return ds

    train_ds = prepare_for_training(train_ds_raw, augment=True)
    test_ds  = prepare_for_training(test_ds_raw, augment=False)

    # ---------- compute class weights ----------
    train_labels_list = [lbl for lbl in train_labels]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array(classes), y=train_labels_list)
    class_weight_dict = {i: w for i,w in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)

    # ---------- main experiment loop ----------
    results = []

    for epochs in EPOCHS_LIST:
        print("\n" + "="*40)
        print(f"RUN for epochs = {epochs}")
        print("="*40)
        # build model and base
        model, base = build_resnet_head(len(classes), dropout_rate=0.5, lr_head=1e-4)

        # determine head-training epochs and fine-tune epochs
        head_epochs = min(HEAD_FREEZE_EPOCHS, epochs)
        ft_epochs = max(0, epochs - head_epochs)

        # Train head (base frozen)
        print(f"Training head for {head_epochs} epochs (base frozen)...")
        t0 = time.time()
        model.fit(train_ds, validation_data=test_ds, epochs=head_epochs, class_weight=class_weight_dict, verbose=1)
        head_time = time.time() - t0

        # Fine-tune if needed
        ft_time = 0.0
        if ft_epochs > 0:
            print(f"Fine-tuning: unfreezing last layers of base and training for {ft_epochs} epochs...")
            # Unfreeze last convolutional block of ResNet50: unfreeze from layer index -50 onward or use layer name
            # Simpler: unfreeze the whole base then re-freeze early layers to reduce training cost
            base.trainable = True
            # freeze early layers
            fine_tune_at = len(base.layers) - 30  # unfreeze last 30 layers
            for i, layer in enumerate(base.layers):
                layer.trainable = i >= fine_tune_at
            # compile with lower LR
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            t0 = time.time()
            model.fit(train_ds, validation_data=test_ds, epochs=ft_epochs, class_weight=class_weight_dict, verbose=1)
            ft_time = time.time() - t0

        total_train_time = round(head_time + ft_time, 2)

        # Evaluate CNN on test set
        cnn_test_acc = float(model.evaluate(test_ds, verbose=0)[1])
        print(f"Finished CNN training: test accuracy = {cnn_test_acc:.4f}, train time = {total_train_time}s")

        # Build feature extractor (outputs the Dense(128) penultimate features)
        feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('feat_dense').output)

        # Extract features for train & test datasets (use raw preprocessed datasets)
        print("Extracting features for train...")
        X_train, y_train = extract_features_from_model(feature_model, train_ds)
        print("Extracting features for test...")
        X_test, y_test = extract_features_from_model(feature_model, test_ds)

        print("Feature shapes:", X_train.shape, X_test.shape)

        # Train XGBoost on train features
        print("Training XGBoost on extracted features...")
        xgb_clf = xgb.XGBClassifier(**XGB_PARAMS)
        t0 = time.time()
        xgb_clf.fit(X_train, y_train)
        xgb_train_time = time.time() - t0

        # Evaluate XGBoost on test features
        y_pred = xgb_clf.predict(X_test)
        xgb_acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        print(f"XGBoost test acc = {xgb_acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

        # Save models & artifacts
        prefix = f"resnet_ep{epochs}"
        cnn_path = os.path.join(WORK_DIR, f"cnn_{prefix}.h5")
        feat_path = os.path.join(WORK_DIR, f"featmod_{prefix}.h5")
        xgb_path = os.path.join(WORK_DIR, f"xgb_{prefix}.joblib")

        model.save(cnn_path)
        feature_model.save(feat_path)
        joblib.dump(xgb_clf, xgb_path)

        # Save confusion matrix figure
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix (epochs={epochs})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i,j], ha='center', va='center', color='white')
        plt.tight_layout()
        plt.savefig(os.path.join(WORK_DIR, f"confusion_{prefix}.png"))
        plt.close(fig)

        # Save numeric results
        row = {
            "epochs": epochs,
            "cnn_test_acc": cnn_test_acc,
            "cnn_train_time_s": total_train_time,
            "xgb_test_acc": xgb_acc,
            "xgb_train_time_s": round(xgb_train_time, 2),
            "xgb_prec_macro": float(prec),
            "xgb_rec_macro": float(rec),
            "xgb_f1_macro": float(f1),
            "cnn_path": cnn_path,
            "feat_path": feat_path,
            "xgb_path": xgb_path
        }
        results.append(row)
        pd.DataFrame(results).to_csv(os.path.join(WORK_DIR, "results_summary.csv"), index=False)

    print("\nALL RUNS COMPLETE. Summary saved to:", os.path.join(WORK_DIR, "results_summary.csv"))
