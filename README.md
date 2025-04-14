# Image-classification

📌 Key Features
✅ Built using TensorFlow and Keras Sequential API
🧠 Deep CNN architecture with Conv2D, MaxPooling, Dense, Dropout layers
🔄 Applied data augmentation for better generalization
🧪 Evaluated on a test set with validation accuracy tracking
🖼️ Can make predictions on new, unseen images
🧹 Includes image preprocessing, augmentation, and resizing


🗂️ Dataset Structure
Copy
Edit
dog_cat_dataset/
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   └── dogs/
│   └── test_set/
│       ├── cats/
│       └── dogs/


📊 Model Architecture
Conv2D (32 filters) + MaxPooling + Dropout
Conv2D (64 filters) + MaxPooling + Dropout
Flatten → Dense (128) + Dropout
Dense (64) + Dropout
Dense (32) + Dropout
Output: Dense (1) with Sigmoid activation


🚀 How to Run
Clone this repository.
Install dependencies:
pip install tensorflow scikit-learn numpy
Ensure dataset is structured as shown above.
Run the Python script:
bash
Copy
Edit
python dog_cat_cnn.py
The model will train and predict whether a new image is a dog or cat.

🔧 Tech Stack
Python
TensorFlow / Keras
NumPy
Scikit-learn
OpenCV / PIL (via Keras)

Note:- Dataset is very large so it could be not upload.
