import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.fft import fft
from scipy.stats import entropy
import joblib
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import label_binarize


t = np.linspace(0, 10, 100)
n = len(t)


def generate_coherent_signal():
    A = np.random.uniform(0.8, 1.0)
    omega = np.random.uniform(1.0, 3.0)
    phi = np.random.uniform(0, 2 * np.pi)
    base = A * np.cos(omega * t + phi)
    noise = np.random.normal(0, 0.01, size=n)
    return base + noise

def generate_decohered_signal():
    base = generate_coherent_signal()
    energy = np.random.uniform(0.5, 2.0)
    gamma = 1 / (energy ** 2)
    damping = np.exp(-gamma * t)
    noise = np.random.normal(0, 0.1, size=n)
    return base * damping + noise

def generate_partial_signal():
    base = generate_coherent_signal()
    gamma = np.random.uniform(0.01, 0.1)
    damping = np.exp(-gamma * t)
    noise = np.random.normal(0, 0.02, size=n)
    return base * damping + noise

def extract_fft_features(signal):
    freqs = np.abs(fft(signal))[:n // 2]
    norm_freqs = freqs / np.sum(freqs)
    return [
        np.max(freqs),
        np.mean(freqs),
        np.std(freqs),
        entropy(norm_freqs + 1e-12)
    ]


samples_per_class = 1000
X, y = [], []

for _ in range(samples_per_class):
    X.append(extract_fft_features(generate_coherent_signal()))
    y.append(0)

for _ in range(samples_per_class):
    X.append(extract_fft_features(generate_decohered_signal()))
    y.append(1)

for _ in range(samples_per_class):
    X.append(extract_fft_features(generate_partial_signal()))
    y.append(2)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=["Coherent", "Decohered", "Partial"]))

cv_score = cross_val_score(model, X, y, cv=5)
print(f"\n Cross-Val Accuracy (5-fold): {cv_score.mean():.4f}")


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of FFT Feature Space")
plt.colorbar(label="Class (0=Coherent, 1=Decohered, 2=Partial)")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


cm = confusion_matrix(y_test, y_pred)


class_names = ["Coherent", "Decohered", "Partial"]


plt.figure(figsize=(6, 5))
sns.set(style="whitegrid")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'},
            linewidths=0.5, linecolor='gray')

plt.title("Confusion Matrix: FFT Signal Classification", fontsize=14, pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)

plt.tight_layout()


plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


def classify_signal(signal, model):
    features = extract_fft_features(signal)
    pred = model.predict([features])[0]
    probs = model.predict_proba([features])[0]
    labels = ["Coherent", "Decohered", "Partial"]
    return labels[pred], probs[pred]

colors = ['green', 'red', 'orange']
labels = ['Coherent', 'Decohered', 'Partial']
generators = [generate_coherent_signal, generate_decohered_signal, generate_partial_signal]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Signal Types by Class", fontsize=18, y=1.05)

for i, (gen_func, ax) in enumerate(zip(generators, axes)):
    signal = gen_func()
    pred_label, confidence = classify_signal(signal, model)

    ax.plot(t, signal, color=colors[i])
    ax.set_title(labels[i], fontsize=14, weight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True)

    ax.text(0.02, 0.9, f"Predicted: {pred_label} ({confidence:.2f})",
            transform=ax.transAxes, fontsize=11, color='black',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.tight_layout()
plt.savefig("signal_types_by_class_final.png", dpi=300, bbox_inches='tight')
plt.show()

importances = model.feature_importances_
feature_names = ['FFT Max', 'FFT Mean', 'FFT Std', 'FFT Entropy']


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

plt.figure(figsize=(6, 4))
sns.barplot(data=importance_df, x="Importance", y="Feature", hue="Feature", palette="mako", legend=False)
plt.title("Feature Importances", fontsize=13)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()


joblib.dump(model, "fft_rf_model_3class.pkl")
print(" Model saved as fft_rf_model_3class.pkl")

