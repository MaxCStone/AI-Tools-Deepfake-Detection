import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_trained_model():
    model_path= os.path.join('vgg16', 'best_run_finetuned', 'model_b32_e10_augtrue_fttrue.h5')
    
    if not os.path.exists(model_path):
        print("File not found")
    else:
        model=load_model(model_path)
        
    return model
        
def preprocess_image(image_path):
    #load and scale the image
    image = image_utils.load_img(image_path, target_size=(224, 224))
    
    #make image into an array
    image = image_utils.img_to_array(image)
    
    #reshape the image to work with the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    return image

def main():
    model = load_trained_model()
    image=preprocess_image('Resized_Gemini_Pics/11.jpg')
    prediction=model.predict(image)
        
    class_labels = {0: "Fake", 1: "Real"}
    predicted_class = int(prediction[0][0] > 0.5)
    result = class_labels[predicted_class]

    print(f"Prediction: {result}")    

if __name__ == '__main__':
    main()