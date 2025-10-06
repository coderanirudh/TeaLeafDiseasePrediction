import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

app = Flask(__name__)

svc = SVC(probability=True)
rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
knn = KNeighborsClassifier(n_neighbors=5)

cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

Categories = ['algal leaf', 'gray light', 'healthy', 'Red leaf spot', 'white spot']
datadir = 'C:/python_hackathon_project/tea sickness dataset/tea_disease'

def preprocess_image(img):
    """Preprocess the input image by resizing and normalizing."""
    img_resized = resize(img, (150, 150, 3)) 
    return img_resized

def train_models():
    """Train the models on the given dataset."""
    flat_data_arr = []
    target_arr = []
    for i in Categories:
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array = imread(os.path.join(path, img))
            img_resized = preprocess_image(img_array)
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)

    x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.2, random_state=42, stratify=target)
    y_train_categorical = to_categorical(y_train, num_classes=5)
    y_test_categorical = to_categorical(y_test, num_classes=5)

   
    svc.fit(x_train, y_train)
    rfc.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn.fit(x_train.reshape(-1, 150, 150, 3), y_train_categorical, epochs=5, batch_size=32)

    print("Model training completed.")

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image file upload, predict disease, and return results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join('tea sickness dataset/uploads', filename)
    file.save(file_path)

    img = imread(file_path)
    img_resized = preprocess_image(img)
    img_flattened = img_resized.flatten().reshape(1, -1)  
    probability_svm = svc.predict_proba(img_flattened)
    probability_rfc = rfc.predict_proba(img_flattened)
    probability_knn = knn.predict_proba(img_flattened)
    cnn_pred = cnn.predict(np.array([img_resized]))

  
    results = {
        'SVM': {Categories[i]: probability_svm[0][i] * 100 for i in range(len(Categories))},
        'RandomForest': {Categories[i]: probability_rfc[0][i] * 100 for i in range(len(Categories))},
        'KNN': {Categories[i]: probability_knn[0][i] * 100 for i in range(len(Categories))},
        'CNN': {Categories[i]: cnn_pred[0][i] * 100 for i in range(len(Categories))}
    }

   
    predicted_class = {
        'SVM': Categories[svc.predict(img_flattened)[0]],
        'RandomForest': Categories[rfc.predict(img_flattened)[0]],
        'KNN': Categories[knn.predict(img_flattened)[0]],
        'CNN': Categories[cnn_pred.argmax()]
    }

    return jsonify({
        'results': results,
        'predicted_class': predicted_class
    })

if __name__ == '__main__':
    train_models()  
    app.run(debug=True)
