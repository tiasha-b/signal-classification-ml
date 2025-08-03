# signal-classification-ml
This project uses machine learning to classify signal types like coherent and noisy signals. It includes signal generation, FFT-based feature extraction, Random Forest classification, and evaluation via PCA, confusion matrix, and feature importance plots.
# Project Title
AI-Powered Signal State Detection via Spectral Features and Ensemble Classification

This project explores how AI can be used to detect and classify the physical state of time-series signals using spectral features and ensemble learning. It focuses on classifying synthetic signals into three categories: Coherent, Decohered, and Partially Decohered.

The signals are generated to simulate different physical behaviors. Coherent signals are smooth and oscillatory, while decohered signals include damping and noise, representing loss of coherence over time. These differences are captured by analyzing the frequency content of the signals using the Fast Fourier Transform (FFT).

From each signal, four key features are extracted: maximum amplitude, mean amplitude, standard deviation, and spectral entropy. These features are chosen to highlight both the structure and randomness present in each signal type.

Note: Noise and randomness have been intentionally introduced in the simulated data to mimic real-world physical uncertainty, making the model more robust and industry-ready. This helps ensure the classifier handles non-ideal, noisy inputs as encountered in practical applications like medical signals, sensors, or quantum systems.

A Random Forest classifier is trained on these features and delivers strong performance while remaining interpretable. The full pipeline includes:

1. Simulating signals for each class: coherent, decohered, and partially decohered

2. Extracting spectral features using the Fourier transform

3. Training a Random Forest model for classification

4. Evaluating the model through cross-validation, PCA visualization, confusion matrix, and feature importance plots

5. Saving the trained model for future use and demonstrating classification on unseen signals

This workflow is fully modular, easy to extend, and can be adapted for real-world tasks. It provides a solid foundation for applying AI to any problem involving time-domain signals.

This approach is not only effective for synthetic examples but is also highly applicable to real-world scenarios such as ECG/EEG signal classification in healthcare, vibration analysis for predictive maintenance, industrial sensor monitoring, and data-driven diagnostics in physics or engineering.



## Installation

Make sure you have Python 3.7 or higher installed.

Install all required libraries by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```


### Project Structure

    ├── signal_classifier.py           # Main script  
    ├── signal_classifier.ipynb        # Interactive Python version (for notebooks)  
    ├── fft_rf_model_3class.pkl        # Trained model file  
    ├── confusion_matrix.png           # Visual output: confusion matrix  
    ├── feature_importance.png         # Visual output: feature importance bar plot  
    ├── signal_types_by_class.png      # Sample classified signal plot  
    └── README.md                      # Project documentation

##  Result Visualizations

To test real-world robustness, noise was intentionally introduced into the signal data. These visualizations illustrate how well the classifier performs even under imperfect, noisy conditions.

---

### 1. Signal Types by Class

Shows example signals from each class after FFT transformation.

![Signal Types](signal_types_by_class_.png)

- This plot visualizes the frequency spectrum for a sample from each of the three signal classes: **Coherent**, **Decohered**, and **Partial**.
- You can see:
  - **Coherent** signals show a sharp, clear frequency peak.
  - **Decohered** signals appear flatter and noisier.
  - **Partial** signals lie in between, often showing some structure with significant noise.
- Helps understand what the model "sees" when making its decision.

---

### 2. PCA Projection of FFT Features

Visualizes the FFT-based features reduced to 2D using PCA.

![PCA FFT Plot](pca_fft_feature_plot.png)

- Principal Component Analysis (PCA) is used here to project high-dimensional FFT features into 2D for visualization.
- We see distinct clustering:
  - **Decohered** signals form a separate, compact cluster.
  - **Coherent** and **Partial** signals show some overlap, which is consistent with the confusion matrix findings.
- Indicates that while the classes are generally separable, some fuzziness exists between Coherent and Partial.



### 3. Feature Importance (Random Forest)

Highlights which FFT features the classifier relies on the most.

![Feature Importance](feature_importance.png)

- This bar plot shows the importance scores assigned to FFT frequency bins by the Random Forest.
- Features around certain frequency ranges are significantly more influential, suggesting that key discriminative information lies in those bands.
- Can be used for feature selection or interpretability.

---

### 4. Confusion Matrix

Shows model predictions versus true labels.

![Confusion Matrix](confusion_matrix.png)

- **Diagonal values** show correct predictions; **off-diagonal** are misclassifications.
- Summary:
  - Coherent signals: 201 correct, 16 misclassified as Partial.
  - Decohered signals: Perfectly classified — 197 correct.
  - Partial signals: 155 correct, 31 misclassified as Coherent.
-  **Insight**: Most confusion is between **Partial** and **Coherent**, likely due to overlapping spectral features.

---



---




