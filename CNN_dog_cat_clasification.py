import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score
import numpy as np

# Initialize the CNN model
cnn = Sequential()

# First Convolutional Block
cnn.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))  # Added dropout

# Second Convolutional Block
cnn.add(Conv2D(64, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))  # Added dropout

# Flatten the layers to connect to dense layers
cnn.add(Flatten())

# First Dense Layer
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))  # Added dropout

# Second Dense Layer
cnn.add(Dense(64, activation='relu'))
cnn.add(Dropout(0.5))  # Added dropout

# Third Dense Layer
cnn.add(Dense(32, activation='relu'))
cnn.add(Dropout(0.5))  # Added dropout

# Output Layer
cnn.add(Dense(1, activation='sigmoid'))

# Compile the model
cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 1.5]
)

# Data preprocessing for test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\ketan\ketan_python\dog_cat_dataset\dataset\training_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    r"C:\Users\ketan\ketan_python\dog_cat_dataset\dataset\test_set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Train the model
cnn.fit(train_generator, steps_per_epoch=500, epochs=50, validation_data=test_generator, validation_steps=800)

# Predict a new image
img = image.load_img(r"C:\Users\ketan\ketan_python\dog_cat_dataset\dataset\training_set\dogs\dog.28.jpg", target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Make prediction
pred = cnn.predict(img)
if pred > 0.5:
    print("Cat")
else:
    print("Dog")

# Accuracy calculation
acc_test = accuracy_score(test_generator, pred) * 100
print(f"Test Accuracy: {acc_test:.2f}%")

