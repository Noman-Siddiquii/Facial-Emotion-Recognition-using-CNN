# Facial Emotion Recognition Using CNN

## Overview
Facial Emotion Recognition (FER) is a key challenge in computer vision, involving the detection of human emotions from facial expressions. Applications range from mental health monitoring to human-computer interaction and surveillance. This project implements a Convolutional Neural Network (CNN) to classify emotions using the **RAF-DB** dataset, which contains over 15,000 labeled facial images across **seven categories**: Happy, Sad, Angry, Fear, Disgust, Surprise, and Neutral.

We conducted thorough preprocessing including normalization, class balancing via augmentation, and one-hot encoding. Our model achieved high accuracy and generalization across classes, making it viable for real-world use cases.

---

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Dataset
We used the **RAF-DB (Real-world Affective Faces Database)** for training and evaluation. It consists of diverse facial expressions annotated with seven emotion classes:
- Happy
- Angry
- Sad
- Fear
- Disgust
- Surprise
- Neutral

---

## Preprocessing
To prepare the dataset for CNN training:
- **Merging**: Combined train and test splits for uniform processing.
- **Shuffling**: Randomized input to reduce learning bias.
- **Normalization**: Scaled pixel values to the [0, 1] range.
- **Label Encoding**: Applied one-hot encoding after adjusting class labels.
- **Data Augmentation & Class Balancing**:
  - Over-represented classes (e.g., *Happy*) were downsampled.
  - Underrepresented classes were augmented using:
    - Rotation
    - Horizontal flip
    - Zoom transformations

---

## Model Architecture
The CNN architecture includes:
- Convolutional layers with **ReLU** activation
- **MaxPooling** for spatial reduction
- Fully connected **Dense layers** with **Dropout**
- **Softmax** output layer for multiclass classification

📷 *Figure 1: CNN Model Architecture*  
![image](https://github.com/user-attachments/assets/4ed33838-8d2a-4ce7-949a-9182cbd84548)


---

## Training Details
- **Epochs**: 60  
- **Batch Size**: 64  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy

📈 *Figure 2: Training/Validation Accuracy and Loss Curves*  
![image](https://github.com/user-attachments/assets/73a3ef85-f55b-4edc-a396-29f365d04092)


---

## Results
- **Training Accuracy**: 84.31%
- **Validation Accuracy**: 83.84%
- **Test Accuracy**: 83.00%

📊 *Figure 3: Classification Report*  
![image](https://github.com/user-attachments/assets/45f28b3d-f082-40ca-aa68-64feca34147a)


📉 *Figure 4: Confusion Matrix*  
![image](https://github.com/user-attachments/assets/0ac6c944-ec0c-49ee-bb3e-eaa1b9fd4103)



📸 *Figure 5: Sample Predictions with Confidence Scores*  
![image](https://github.com/user-attachments/assets/2daa1a57-5f69-47c6-9d6e-19e1fb2c6ced)


Notable insights:
- Model performed well across all classes.
- Augmentation significantly improved minority class accuracy.
- Misclassifications typically occurred in low-light or occluded images.

---

## Conclusion

### ✅ Key Findings
- CNN-based FER system achieved **83% accuracy** on real-world data.
- Class imbalance mitigation improved performance on rare emotions.
- Model generalized well across challenging conditions.

### ⚠ Limitations
- Misclassifications in occluded and mixed-emotion images.
- Performance under varied lighting/backgrounds needs further tuning.
- Transfer learning or ensemble models could enhance future performance.


