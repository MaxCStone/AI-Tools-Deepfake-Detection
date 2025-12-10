

import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score


def process_image(image_path):
    image = load_img(image_path, color_mode="grayscale")
    image_array = img_to_array(image) / 255
    processed_image = image_array[0] + image_array[1]
    return processed_image


real_train_image_path = "data/Dataset/Train/Real/"
fake_train_image_path = "data/Dataset/Train/Fake/"
real_test_image_path = "data/Dataset/Test/Real/"
fake_test_image_path = "data/Dataset/Test/Fake/"


# Data setup:

real_images = []
print(os.getcwd())
for path in os.listdir(real_train_image_path):
    real_images.append(process_image(real_train_image_path + path))
for path in os.listdir(real_test_image_path):
    real_images.append(process_image(real_test_image_path + path))



false_images = []
for path in os.listdir(fake_train_image_path):
    false_images.append(process_image(fake_train_image_path + path))
for path in os.listdir(fake_test_image_path):
    false_images.append(process_image(fake_test_image_path + path))


labels = []
for i in range(len(false_images)):
    labels.append(0)
for i in range(len(real_images)):
    labels.append(1)
all_images = np.asarray(false_images + real_images)
all_images = np.squeeze(all_images)
# Model Setup:

x_train, x_test, y_train, y_test = train_test_split(all_images, labels, test_size=0.2, stratify=labels, random_state=42)

decision_tree = RandomForestClassifier()
decision_tree.fit(x_train, y_train)

# Model Testing and Analysis:

y_pred=decision_tree.predict(x_test)

f = open("accuracy.txt", "w")
f.write(str(accuracy_score(y_test, y_pred)))
f.flush()
f.close()   


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('confusion_matrix.png')
print("Random Forest Model Complete")

