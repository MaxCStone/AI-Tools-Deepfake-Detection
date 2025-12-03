"""
    will create the model that will be used in the juypter notebook
"""

import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--main_dir', type=str, default='')
    parser.add_argument('--augment_data', type=str, default='')
    parser.add_argument('--fine_tune', type=str, default='')
    return parser.parse_args()

def main():
    args = parse_args()
    data = args.data
    my_batch_size = int(args.batch_size)
    my_epochs = int(args.epochs)
    augment_data = args.augment_data
    fine_tune = args.fine_tune
    
    output_dir = os.path.join(args.main_dir, 'vgg16')
    os.makedirs(output_dir, exist_ok=True)
    
    h5modeloutput = 'model_b' + args.batch_size + '_e' + args.epochs + '_aug' + \
        args.augment_data + '_ft' + args.fine_tune + '.h5'
    print(args)
    
    base_model = keras.applications.VGG16(
        weights='imagenet',  
        input_shape=(224, 224, 3),
        include_top=False)

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)

    x =  keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer='Adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

    datagen = ImageDataGenerator(
            samplewise_center=True,  
            rotation_range=0,  
            zoom_range = 0,  
            width_shift_range=0,  
            height_shift_range=0,  
            horizontal_flip=False,  
            vertical_flip=False) 

    if(augment_data.lower() in ['true', '1', 't', 'y', 'yes']):
        datagen = ImageDataGenerator(
                samplewise_center=True,
                rotation_range=0,  
                zoom_range = 0.01, 
                width_shift_range=0.01,  
                height_shift_range=0.01, 
                horizontal_flip=False,  
                vertical_flip=False) 

    train_it = datagen.flow_from_directory( '../data/Dataset/Train/', 
                                           target_size=(224,224), 
                                           color_mode='rgb', 
                                           batch_size=my_batch_size,
                                           class_mode="binary")
    valid_it = datagen.flow_from_directory('../data/Dataset/Validation/', 
                                          target_size=(224,224), 
                                          color_mode='rgb', 
                                           batch_size=my_batch_size,
                                          class_mode="binary",
                                          shuffle=False)

    history_object = model.fit(train_it,
              validation_data=valid_it,
              steps_per_epoch=int(train_it.samples/train_it.batch_size),
              validation_steps=int(valid_it.samples/valid_it.batch_size),
              epochs=my_epochs,
              verbose=2)

    if(fine_tune.lower() in ['true', '1', 't', 'y', 'yes']):
        base_model.trainable = True
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
                      loss =  'binary_crossentropy' , metrics = ['accuracy'])

        history_object = model.fit(train_it,
                  validation_data=valid_it,
                  steps_per_epoch=int(train_it.samples/train_it.batch_size),
                  validation_steps=int(valid_it.samples/valid_it.batch_size),
                  epochs=my_epochs)
                  
    save_loss_plot(history_object.history, args, output_dir)

    def save_confusion_matrix(model, valid_it, args, output_dir):
        # Ensure generator yields in a fixed order
        try:
            valid_it.reset()
        except Exception:
            pass

        steps = int(np.ceil(valid_it.samples / valid_it.batch_size))
        preds = model.predict(valid_it, steps=steps, verbose=0)
        y_pred = (preds.ravel() > 0.5).astype(int)
        # DirectoryIterator exposes the true classes in `classes`
        y_true = valid_it.classes[:len(y_pred)]

        cm = confusion_matrix(y_true, y_pred)
        labels = list(valid_it.class_indices.keys())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        cm_filename = 'confusion_b' + args.batch_size + '_e' + args.epochs + '.png'
        cm_path = os.path.join(output_dir, cm_filename)
        plt.savefig(cm_path)
        plt.clf()

        report = classification_report(y_true, y_pred, target_names=labels)
        report_path = os.path.join(output_dir, 'classification_report_b' + args.batch_size + '_e' + args.epochs + '.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Saved confusion matrix to {cm_path} and classification report to {report_path}")

    # Save confusion matrix for validation data
    try:
        save_confusion_matrix(model, valid_it, args, output_dir)
    except Exception as e:
        print(f"Failed to create confusion matrix: {e}")

    # Try saving as HDF5 (requires h5py). If that fails, fall back to
    # TensorFlow SavedModel format which doesn't require h5py.
    h5_path = os.path.join(output_dir, h5modeloutput)
    try:
        model.save(h5_path)
        print(f"Model saved to HDF5 file: {h5_path}")
    except Exception as e:
        print(f"HDF5 save failed: {e}\nFalling back to TensorFlow SavedModel format.")
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        model.save(saved_model_dir)
        print(f"Model saved in SavedModel format at: {saved_model_dir}")

    model.summary()


def save_loss_plot(history, args, output_dir):
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Fruit Classification, Batch Size: ' + args.batch_size + ' Epochs: ' + args.epochs)
    plt.legend()
    
    plot_filename= 'model_b' + args.batch_size + '_e' + args.epochs + '.png'
    plot_path=os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)

if __name__ == "__main__":
    main()