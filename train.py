import os
import librosa
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "../dataset"

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

CLASSES = [
    "car_horn",
    "emergency_siren",
    "human_scream",
    "door_knock",
    "gunshot"
]


# ==========================================
# LOAD AUDIO AND EXTRACT MFCC
# ==========================================

def extract_mfcc(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Make every audio sample exactly 3 seconds
    required_length = SAMPLE_RATE * DURATION

    if len(audio) < required_length:
        audio = np.pad(
            audio,
            (0, required_length - len(audio))
        )
    else:
        audio = audio[:required_length]

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    return mfcc


# ==========================================
# LOAD DATASET
# ==========================================

X = []
y = []

for class_name in CLASSES:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for filename in os.listdir(class_path):

        if filename.endswith(".wav"):

            file_path = os.path.join(
                class_path,
                filename
            )

            try:

                mfcc = extract_mfcc(file_path)

                X.append(mfcc)
                y.append(class_name)

                print(
                    f"Loaded: {class_name}/{filename}"
                )

            except Exception as e:

                print(
                    f"Error loading {file_path}: {e}"
                )


# Convert to NumPy arrays

X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)
print("Labels:", y.shape)


# ==========================================
# ENCODE LABELS
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nClasses:")
for i, class_name in enumerate(encoder.classes_):
    print(i, "=", class_name)


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==========================================
# CNN INPUT SHAPE
# ==========================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]


# ==========================================
# BUILD CNN
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=X_train.shape[1:]
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        len(CLASSES),
        activation="softmax"
    )
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ==========================================
# TRAIN
# ==========================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=32
)


# ==========================================
# FINAL TEST
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("../models", exist_ok=True)

model.save(
    "../models/alertlens_cnn.keras"
)

print("\nModel saved!")