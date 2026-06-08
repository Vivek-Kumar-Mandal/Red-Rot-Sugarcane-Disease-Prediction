import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Reshape, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = 200
# Use repository-relative dataset folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CATEGORIES = ["Healthy", "Unhealthy"]
training_data = []


def create_training_data(dataset_dir=DATASET_DIR):
    for category in CATEGORIES:
        path = os.path.join(dataset_dir, category)
        class_num = CATEGORIES.index(category)

        # Check if path exists
        if not os.path.exists(path):
            print(f"Directory '{path}' not found. Skipping {category}.")
            continue

        for img in os.listdir(path):
            try:
                image_path = os.path.join(path, img)
                img_array = cv2.imread(image_path)

                # Ensure the image was read successfully
                if img_array is None:
                    print(f"Warning: Unable to read {image_path}")
                    continue

                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_array, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")


def build_and_train(epochs=10, batch_size=32):
    create_training_data()

    # Shuffle and separate features and labels
    np.random.shuffle(training_data)
    X = [features for features, label in training_data]
    y = [label for features, label in training_data]

    # Convert to numpy arrays and normalize
    if len(X) == 0:
        raise SystemExit("No training images found. Place dataset in the 'dataset/' folder with expected subfolders.")

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    y = np.array(y)

    # One-hot encode labels
    y = to_categorical(y, num_classes=len(CATEGORIES))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the CNN-RNN hybrid model
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(1024, activation='relu'),
        Reshape((8, 128)),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(CATEGORIES), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score: ", score[0])
    print("Test Accuracy: ", score[1])

    return model


def predict(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read {image_path}")
        return None

    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_data = np.expand_dims(resized_img, axis=0) / 255.0

    prediction = model.predict(input_data)
    class_index = np.argmax(prediction[0])
    return CATEGORIES[class_index]


if __name__ == "__main__":
    model = build_and_train()

    # Example usage (repository-relative)
    EXAMPLE_IMAGE = os.path.join(DATASET_DIR, 'Healthy', 'example.jpg')
    if os.path.exists(EXAMPLE_IMAGE):
        try:
            print('Predicting on example image:', EXAMPLE_IMAGE)
            print('Predicted class:', predict(EXAMPLE_IMAGE, model))
        except Exception:
            pass
