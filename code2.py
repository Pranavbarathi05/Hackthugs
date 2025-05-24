import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

data = []
labels = []

for file in os.listdir('gesture_data'):
    if file.endswith('.npy'):
        label = file.split('_')[0]
        sample = np.load(os.path.join('gesture_data', file))
        data.append(sample)
        labels.append(label)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y = to_categorical(y)
X = np.array(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('labels.npy', encoder.classes_)
