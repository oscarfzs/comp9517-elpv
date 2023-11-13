
from PIL import Image
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# %%
def load_dataset(fname=None, new_size=(10, 10)):
    if fname is None:
        # Assume we are in the utils folder and get the absolute path to the
        # parent directory.
        base_dir = os.getcwd()
        fname = os.path.join(base_dir, '../elpv-dataset/labels.csv')

    data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[
                         'path', 'probability', 'type'])
    image_fnames = np.char.decode(data['path'])
    probs = data['probability']
    types = np.char.decode(data['type'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            image = image.convert('L')  # Convert to grayscale
            image = image.resize(new_size, Image.Resampling.LANCZOS)  # Resize the image
            return np.asarray(image)

    dir = os.path.dirname(fname)

    images = np.array([load_cell_image(os.path.join(dir, fn))
                       for fn in image_fnames])

    return images, probs, types

images, proba, types = load_dataset(new_size=(10, 10))  


N = 5
plt.figure(figsize=(15, 10))
for i in range(N):
    ax = plt.subplot(1, N, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Prob: {proba[i]}, Type: {types[i]}")
    plt.axis('off')
plt.show()



# %%
# Flatten the images for ML algorithms
X = images.reshape((images.shape[0], -1))
y = (proba > 0.5).astype(int)  # Convert probabilities to binary labels for classification

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# %%
# Create a pipeline with PCA and a classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=150)),  # Adjust the number of components
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameters for grid search
param_grid = {
    'pca__n_components': [50, 100, 150],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}


# %%
# Grid search with cross-validation
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)


# %%
# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Plotting confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 打印准确度
print(f"Accuracy on Test Set: {accuracy:.2f}")


