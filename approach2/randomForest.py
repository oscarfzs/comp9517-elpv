import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_dataset(fname=None):
    # Set the default file path for the dataset if none is provided
    if fname is None:
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        fname = os.path.join(fname, 'elpv-dataset/labels.csv')

    # Load the data from the CSV file
    data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type'])
    image_fnames = np.char.decode(data['path'])
    probs = data['probability']
    types = np.char.decode(data['type'])

    # Define a function to load an image
    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    # Get the directory of the images
    dir = os.path.dirname(fname)

    # Load all images into an array
    images = np.array([load_cell_image(os.path.join(dir, fn)) for fn in image_fnames])
    
    # Return the loaded images, probabilities, and types
    return images, probs, types


# Load the dataset
images, probs, types = load_dataset()

# Flatten the images from 3D to 2D for traditional machine learning models
n_samples = len(images)
X = images.reshape((n_samples, -1))  # '-1' means the remaining dimensions are automatically calculated

# Encode the labels, mapping 'mono' to 0 and 'poly' to 1
type_to_num = {'mono': 0, 'poly': 1}
y = np.array([type_to_num[type_str] for type_str in types])

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the Random Forest classifier model
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.2%}')
