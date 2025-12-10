from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import argparse
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--main_dir', type=str, default='')
    parser.add_argument('--augment_data', type=str, default='')
    parser.add_argument('--fine_tune', type=str, default='')
    return parser.parse_args()

def load_data(data_path, subset):
    subset_path = os.path.join(data_path, subset)
    labels=[]
    images=[]
    
    for class_idx, class_name in enumerate(os.listdir(subset_path)):
        class_path = os.path.join(subset_path, class_name) #should be gettting specifically real/fake
        
        for img_file in os.listdir(class_path):
            img_path=os.path.join(class_path, img_file)
            img = Image.open(img_path).resize((64,64))
            img_array = np.array(img).flatten()
            
            images.append(img_array)
            labels.append(class_idx)
    
    return np.array(images), np.array(labels)

def save_confusion_matrix(model, X, y, filename):
    y_pred= model.predict(X)
    accuracy=accuracy_score(y, y_pred)
    cm= confusion_matrix(y, y_pred)
    disp= ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    print(f"Accuracy Score {X}: {accuracy}")
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.clf()

def save_accuracy(filename, content, mode='w'):
    try:
        with open(filename, mode) as f:
            if isinstance(content, list):
                f.writelines(f'{line}\n' for line in content)
            else:
                f.write(str(content)) 
    except OSError as e:
        print("Flie Error: ", e)
        
def main():
    data = '../data/Dataset'
    
    #returns numpy array of the images and labels
    train_X, train_y=load_data(data, 'Train')
    val_X, val_y=load_data(data, 'Validation')
    test_X, test_y=load_data(data, 'Test')
    
    pipe = make_pipeline(
        StandardScaler(), 
        LogisticRegression(max_iter=1000))
    pipe.fit(train_X,train_y)
    
    train_acc=pipe.score(train_X, train_y)
    val_acc=pipe.score(val_X, val_y)
    test_acc=pipe.score(test_X, test_y)
    
    content = f'Training Acc: {train_acc}\nValidation Acc: {val_acc}\nTesting Acc: {test_acc}'
    save_accuracy("overall_acc.txt", content)
    
    save_confusion_matrix(pipe, val_X, val_y, 'confusion_matrix_val.png')
    save_confusion_matrix(pipe, test_X, test_y, 'confusion_matrix_test.png')
    
if __name__ == "__main__":
    main()
    