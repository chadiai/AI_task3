import streamlit as st
import tensorflow as tf
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import image_dataset_from_directory

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

# Constants
NUM_CLASSES = 5
IMG_SIZE = 64
HEIGHT_FACTOR = 0.1
WIDTH_FACTOR = 0.1
BATCH_SIZE = 32
EPOCHS = 20

# Set the parameters for your data
image_size = (IMG_SIZE, IMG_SIZE)
validation_split = 0.2

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import ModelCheckpoint
from keras.utils import image_dataset_from_directory

def perform_eda(dataset_dir, num_samples_to_visualize):
    st.header('Exploratory Data Analysis')

    # List the subfolders (classes) in the dataset directory
    classes = os.listdir(dataset_dir)

    # Initialize a dictionary to store the image counts for each class
    image_counts = {cls: 0 for cls in classes}

    # Initialize a dictionary to store a few sample image paths from each class
    sample_images = {cls: [] for cls in classes}

    # Loop through each class and perform EDA
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        class_images = os.listdir(class_dir)

        # Count the number of images in each class
        image_counts[cls] = len(class_images)

        # Randomly select a few sample images
        sample_images[cls] = random.sample(class_images, num_samples_to_visualize)
      # Print the image counts for each class

    images_row = st.columns(5)
    for i, (cls, images) in enumerate(sample_images.items()):
        if images:
            image_name = images[0]  # Display the first image for each class
            image_path = os.path.join(dataset_dir, cls, image_name)
            img = Image.open(image_path)

            # Display the image and its caption in the same column
            with images_row[i]:
                st.image(img, use_column_width=True)
                st.text(cls)
        else:
            # Display a message if no images are available for the class
            with images_row[i]:
                st.text(f"No images available for {cls}")
    col1, col2, col3 = st.columns(3)
    with col2:
        # Use a button to reload images
        if st.button("Randomize Image preview",):
            pass
    for cls, count in image_counts.items():
        st.write(f"{cls} images count: {count}")


def build_and_train_model(train_ds, validation_ds, num_epochs, learning_rate):
    # Model Definition
    model = tf.keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(HEIGHT_FACTOR, WIDTH_FACTOR),
        
        layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),


        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Create a callback to save the best model during training
    model_checkpoint_callback = ModelCheckpoint(
        filepath='wild_cat.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Compile and train your model as usual
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_ds,
                            validation_data=validation_ds,
                            epochs=num_epochs,
                            callbacks=[model_checkpoint_callback],
                            verbose=1)
    return history, model

# Sidebar Configuration
st.sidebar.title('Train the model')
dataset_dir = './dataset/train/'
testset_dir = './dataset/test/'

classes = os.listdir(dataset_dir)

# Perform EDA
perform_eda(dataset_dir, num_samples_to_visualize=5)

# Sidebar Configuration for Training
num_epochs = st.sidebar.slider('Number of Epochs', min_value=1, max_value=30, value=10)
learning_rate = st.sidebar.number_input('Learning Rate', min_value=0.001, max_value=1.0, value=0.001, step=0.001, format="%.3f")

# Load Datasets
train_ds = image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

validation_ds = image_dataset_from_directory(
    directory=dataset_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

test_ds = image_dataset_from_directory(
        directory=testset_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=image_size
)

# Model Training
if st.sidebar.button('Start Training'):
    with st.spinner('Training in progress...'):
        history, model = build_and_train_model(train_ds, validation_ds, num_epochs, learning_rate)
    st.success('Training completed.')  # Display a success message when training is complete
    
    st.subheader('Loss and Accuracy curves')
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()
    st.pyplot(fig)
    
    st.subheader('Confusion matrix')
    # Load and display the confusion matrix
    true_labels = []
    predicted_labels = []

    # Iterate through the validation dataset and make predictions
    for batch in validation_ds:
        images, labels = batch
        true_labels.extend(np.argmax(labels, axis=1))  # Convert one-hot encoded labels to integers
        predictions = model.predict(images)
        predicted_labels.extend(np.argmax(predictions, axis=1))

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots()

    # Display the confusion matrix using the Streamlit `st.pyplot()` function
    class_labels = train_ds.class_names
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    display.plot(cmap='viridis', values_format='d', ax=ax)
    st.pyplot(fig)