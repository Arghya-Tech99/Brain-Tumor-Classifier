
# Brain Tumor Detection Using Deep Learning & Computer Vision

This project is an end-to-end Deep Learning application designed to detect and classify brain tumors from MRI scan images. By leveraging Transfer Learning and Computer Vision, the system can identify four distinct categories of scans with high accuracy through a user-friendly web interface.

## Overview
Early diagnosis of brain tumors is critical for effective treatment. This project automates the classification process using a VGG16 architecture, providing a reliable second opinion for medical professionals.

### Key features
- **Multiclass classification:** Categorizes MRI scans into one of the four classes: Glioma, Meningioma, Pituitary Tumor and No Tumor
- **Deep Learning Pipeline:** Includes image preprocessing, data augmentation, and transfer learning.
- **Web Interface:** A Flask-based web application allowing users to upload MRI images and receive real-time predictions.
- **Performance Visualization:** Includes Accuracy/Loss plots, Confusion Matrix, and ROC Curves for model evaluation.

## Technical Stack
- **Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Frontend and Backend:** Flask, HTML, CSS
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, Tensorflow and Pillow
## Methodology

### 1. Data Preprocessing & Augmentation
To improve model generalization, the following were applied:
- Resizing: All images standardized to a fixed input size for VGG16.
- Normalization: Scaling pixel values to a range of [0, 1].
- Augmentation: Rotation, flipping, and contrast enhancement to expand the dataset diversity.

### 2. Model Architecture
Transfer learning utilized with VGG16
- The base layers are pre-trained on the ImageNet dataset.
- Custom fully connected layers were added to the top to adapt the model to our specific 4-class medical task.

### 3. Model evaluation
The model's performance was rigorously tested using:
- Confusion Matrix: To identify specific class misclassifications.
- ROC Curve: To measure the true positive rate vs. false positive rate across different thresholds.

## Installation and Usage

### 1. Clone the repository
```bash
git clone [https://github.com/Arghya-Tech99/Brain-Tumor-Classifier.git](https://github.com/Arghya-Tech99/Brain-Tumor-Classifier.git)
cd kidney-disease-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main flask application
```bash
python main.python
```
## Future ideas and improvements
- **Explainable AI (XAI):** Integrate Grad-CAM to generate heatmaps that highlight the specific areas of the MRI scan the model used to make its decision. This builds trust with medical practitioners.
- **Advanced Architectures:** Experiment with EfficientNet or Vision Transformers (ViT) to potentially achieve higher accuracy with fewer parameters.
- **Cloud Deployment:** Deploy the application using Docker on platforms like AWS, Azure, or Google Cloud for global accessibility.
- **API Development:** Build a RESTful API using FastAPI to allow integration with hospital management systems.
- **Segmented Masking:** Move beyond classification to Image Segmentation (using U-Net) to calculate the exact volume and dimensions of the tumor.
## Authors

- [@Arghya-Tech99](https://github.com/Arghya-Tech99)

