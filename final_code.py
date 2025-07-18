import cv2
import numpy as np
from PIL import Image
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# Apply Gabor filter for texture analysis
def apply_gabor_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_response, _ = gabor(image, frequency=0.9)
    gabor_normalized = cv2.normalize(gabor_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gabor_normalized

# The segment function segments the pest region based on color characteristics using HSV color space:
def segment_pest(image_path):
    im = Image.open(image_path)
    im_array = np.array(im)
    if len(im_array.shape) == 2 or im_array.shape[2] == 1:
        hsv_image = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_RGB2HSV)
    else:
        hsv_image = cv2.cvtColor(im_array, cv2.COLOR_RGB2HSV)

    min_val, max_val = hsv_image.min(axis=(0, 1)), hsv_image.max(axis=(0, 1))
    mean_val, stddev_val = hsv_image.mean(axis=(0, 1)), hsv_image.std(axis=(0, 1))
    lower_bound = np.clip(mean_val - (1.0 * stddev_val), min_val, max_val)
    upper_bound = np.clip(mean_val + (1.0 * stddev_val), min_val, max_val)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(im_array, im_array, mask=mask)
    return segmented_image

""" The post_process function cleans the segmented pest image using binary thresholding and morphological operations
otsu's method for thresholding and converting it to binary image ':"""
def post_process(segmented_image):
    if len(segmented_image.shape) == 2:
        gray = segmented_image
    else:
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

""" The extract_features function extracts both color and texture features to build a feature vector for each image
colour histogram and GLCM is used, A GLCM is computed on the Gabor-filtered image to capture texture features. 
Four metrics are extracted: contast, dissimilarity, homogenity, ASM((Angular Second Moment))"""
def extract_features(image_path):
    segmented_image = segment_pest(image_path)
    cleaned_image = post_process(segmented_image)
    gabor_image = apply_gabor_filter(cleaned_image)

    if len(segmented_image.shape) == 2 or segmented_image.shape[2] == 1:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)

    hist = cv2.calcHist([segmented_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    glcm = graycomatrix(gabor_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]

    features = np.hstack([hist, contrast, dissimilarity, homogeneity, asm])
    return features, segmented_image, cleaned_image, gabor_image

""" Random forest classification is used in which a matrix is made for dataset, each row represent one image and 
eacch column represent one feature, while extracting the features one single feature vector was created which is 
used for classification"""
def load_and_train_model(dataset_folder):
    labels, features = [], []
    label_encoder = LabelEncoder()

    for folder_name in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in tqdm(os.listdir(folder_path), desc=f"Processing {folder_name}"):
                image_path = os.path.join(folder_path, filename)
                if os.path.isfile(image_path):
                    feature_vector, _, _, _ = extract_features(image_path)
                    features.append(feature_vector)
                    labels.append(folder_name)

    X = np.array(features)
    y = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42) 
    """100 decesion trees each built on a 
random subset of the training data Each tree learns patterns in the data and votes on the pest category 
for any given feature vector."""
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    return classifier, label_encoder

# Predict pest category for images in a test folder
def predict_images(classifier, label_encoder, test_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for test_image in os.listdir(test_folder):
        test_image_path = os.path.join(test_folder, test_image)
        if os.path.isfile(test_image_path):
            feature_vector, segmented_image, cleaned_image, gabor_image = extract_features(test_image_path)
            predicted_label = classifier.predict(feature_vector.reshape(1, -1))
            predicted_class = label_encoder.inverse_transform(predicted_label)[0]
            print(f"The predicted pest category for {test_image} is: {predicted_class}")

            # Create output sub-folder if it does not exist
            class_output_folder = os.path.join(output_folder, predicted_class)
            if not os.path.exists(class_output_folder):
                os.makedirs(class_output_folder)

            # Save processed images in the corresponding folder
            base_filename = os.path.splitext(test_image)[0]
            cv2.imwrite(os.path.join(class_output_folder, f'{base_filename}_segmented.jpg'), segmented_image)
            cv2.imwrite(os.path.join(class_output_folder, f'{base_filename}_cleaned.jpg'), cleaned_image)
            cv2.imwrite(os.path.join(class_output_folder, f'{base_filename}_gabor.jpg'), gabor_image)

# Main execution
dataset_folder = r"G:\Saransh\DIP project\farm_insects\farm_insects"
classifier, label_encoder = load_and_train_model(dataset_folder)

test_folder = r"G:\Saransh\DIP project\test_dataset"  # Update with your test image folder path
output_folder = r"G:\Saransh\DIP project\output_images"  # Folder to save output images
predict_images(classifier, label_encoder, test_folder, output_folder)
