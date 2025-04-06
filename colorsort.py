# training_script.py
import os
import numpy as np
import pandas as pd
import torch
import cv2
import joblib
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import ViTModel, ViTImageProcessor, SegformerFeatureExtractor, SegformerModel
from tqdm import tqdm
import imgaug.augmenters as iaa
from skimage.feature import graycomatrix, graycoprops  # For GLCM features try to install ning amu ni na import "pip install scikit-image"

# Paths
base_path = r"C:\Users\JUSTINE\Documents\PROJECTS\Python\Vision Transformer" #change mo lng sa path name ng project environment
dataset_path = os.path.join(base_path, "dataset")
csv_save_path = os.path.join(base_path, "csv_files")
model_save_path = os.path.join(base_path, "models", "knn_model.pkl")

# Create directories if they don't exist
os.makedirs(csv_save_path, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Expected rice varieties
VALID_LABELS = {'RC 64', 'RC 160', 'RC 440'}

image_paths = []
labels = []

# Load dataset
print("Loading dataset...")
for label in os.listdir(dataset_path):
    if label not in VALID_LABELS:
        print(f"Warning: Unknown label '{label}' found in dataset folder")
        continue
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(label_path, img_file))
                labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

if len(image_paths) == 0:
    raise ValueError("No valid images found in the dataset directory!")

print(f"Found {len(image_paths)} images across {len(set(labels))} rice varieties")

# Data augmentation
aug = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),
    iaa.Sometimes(0.5, iaa.ContrastNormalization((0.75, 1.25)))
])

def preprocess_images(image_paths, batch_size=32, augment=False):
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    segmentation_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    segmentation_model = SegformerModel.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512').to(device)
    
    images = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Preprocessing images"):
        batch_paths = image_paths[i:i + batch_size]
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                if augment:
                    img_np = aug.augment_image(img_np)
                inputs = segmentation_extractor(images=Image.fromarray(img_np), return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = segmentation_model(**inputs)
                segmentation_mask = torch.argmax(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

                segmentation_mask_resized = Image.fromarray(segmentation_mask.astype(np.uint8)).resize(img.size, Image.NEAREST)
                segmentation_mask_resized = np.array(segmentation_mask_resized)

                img_processed = img_np * (segmentation_mask_resized[..., np.newaxis] > 0)
                img_tensor = feature_extractor(images=img_processed, return_tensors="pt")
                images.append(img_tensor['pixel_values'][0])
            except Exception as e:
                print(f"Error loading image {path}: {e}")
    
    if not images:
        raise ValueError("No images were processed successfully!")
    return torch.stack(images)

def extract_additional_features(image_paths):
    features_list = []
    for path in tqdm(image_paths, desc="Extracting additional features"):
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("Image could not be loaded")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Existing features
            contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shape_feature = len(contours)
            size_feature = cv2.contourArea(max(contours, key=cv2.contourArea)) if contours else 0
            texture_feature = np.std(img_gray)
            mean_color = cv2.mean(img)[:3]
            color_feature = np.mean(mean_color)
            contrast = np.max(img_gray) - np.min(img_gray)
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-6))
            chalkness_feature = (contrast + entropy) / 2

            # New features
            # 1. Aspect Ratio
            aspect_ratio = 0
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h != 0 else 0

            # 2. Perimeter
            perimeter = cv2.arcLength(largest_contour, True) if contours else 0

            # 3. Color Histogram Features (HSV)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([img_hsv], [0], None, [180], [0, 180]).flatten()  # Hue
            hist_s = cv2.calcHist([img_hsv], [1], None, [256], [0, 256]).flatten()  # Saturation
            hue_mean = np.mean(hist_h * np.arange(180)) / (np.sum(hist_h) + 1e-6)
            saturation_mean = np.mean(hist_s * np.arange(256)) / (np.sum(hist_s) + 1e-6)

            # 4. Chalkiness Distribution
            _, chalky_mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)  # Threshold for white regions
            chalky_area = np.sum(chalky_mask == 255)
            total_area = np.sum(segmentation_mask_resized > 0) if 'segmentation_mask_resized' in locals() else img_gray.size
            chalkiness_ratio = chalky_area / (total_area + 1e-6)

            # 5. GLCM Texture Features
            glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
            glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            # 6. Edge Density
            edges = cv2.Canny(img_gray, 100, 200)
            edge_density = np.sum(edges == 255) / (img_gray.size + 1e-6)

            features_list.append([
                shape_feature, size_feature, texture_feature, color_feature, chalkness_feature,
                aspect_ratio, perimeter, hue_mean, saturation_mean, chalkiness_ratio,
                glcm_contrast, glcm_homogeneity, edge_density
            ])
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            features_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array(features_list)

# Main execution
print("Starting training...")
try:
    # Preprocess all images (with augmentation for training data)
    all_images = preprocess_images(image_paths, augment=True).to(device)
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    
    def extract_features(images):
        with torch.no_grad():
            images = images.to(device)
            features = vit_model(images).last_hidden_state[:, 0, :]
        return features.cpu().numpy()

    # Extract features
    additional_features = extract_additional_features(image_paths)
    features_df = pd.DataFrame(additional_features, columns=[
        'Shape', 'Size', 'Texture', 'Color', 'Chalkness',
        'AspectRatio', 'Perimeter', 'HueMean', 'SaturationMean', 'ChalkinessRatio',
        'GLCM_Contrast', 'GLCM_Homogeneity', 'EdgeDensity'
    ])
    features_df['Label'] = labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(all_images, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images")

    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)

    # Normalize features
    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)

    # Hyperparameter tuning for KNN
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_features_scaled, y_train)
    best_knn = grid_search.best_estimator_
    print(f"Best KNN parameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_knn.predict(X_test_features_scaled)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Evaluation completed.")

    # Save features and test predictions
    features_file_path = os.path.join(csv_save_path, 'rice_features.csv')
    features_df.to_csv(features_file_path, index=False)
    test_predictions_df = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})
    test_predictions_file_path = os.path.join(csv_save_path, 'rice_test_predictions.csv')
    test_predictions_df.to_csv(test_predictions_file_path, index=False)
    print("Test predictions saved to rice_test_predictions.csv")

    # Predict on all images with confidence
    print("Generating predictions for all images...")
    all_features = extract_features(all_images)
    all_features_scaled = scaler.transform(all_features)
    all_predictions = best_knn.predict(all_features_scaled)
    all_distances, _ = best_knn.kneighbors(all_features_scaled, n_neighbors=best_knn.n_neighbors)
    all_confidences = [100 - np.mean(distances) for distances in all_distances]

    # Create DataFrame with image paths, true labels, predictions, and confidence
    all_predictions_df = pd.DataFrame({
        'Image Path': image_paths,
        'True Label': labels,
        'Predicted Label': all_predictions,
        'Confidence (%)': all_confidences
    })
    
    # Save to Excel
    all_predictions_file_path = os.path.join(csv_save_path, 'rice_all_predictions.xlsx')
    all_predictions_df.to_excel(all_predictions_file_path, index=False)
    print(f"All predictions with confidence saved to {all_predictions_file_path}")

    # Save model and scaler
    joblib.dump(best_knn, model_save_path)
    joblib.dump(scaler, os.path.join(base_path, "models", "scaler.pkl"))
    print(f"Model saved to {model_save_path}")
    print(f"Scaler saved to {os.path.join(base_path, 'models', 'scaler.pkl')}")

except Exception as e:
    print(f"Training failed: {e}")