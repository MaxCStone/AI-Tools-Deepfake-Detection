"""
    will create the model that will be used in the juypter notebook
"""

import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib as plt

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

    outputs = keras.layers.Dense(6, activation = 'softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(optimizer='Adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

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

    train_it = datagen.flow_from_directory( data + '/train/', 
                                           target_size=(224,224), 
                                           color_mode='rgb', 
                                           batch_size=my_batch_size,
                                           class_mode="categorical")
    valid_it = datagen.flow_from_directory( data + '/test/', 
                                          target_size=(224,224), 
                                          color_mode='rgb', 
                                           batch_size=my_batch_size,
                                          class_mode="categorical")

    history_object = model.fit(train_it,
              validation_data=valid_it,
              steps_per_epoch=train_it.samples/train_it.batch_size,
              validation_steps=valid_it.samples/valid_it.batch_size,
              epochs=my_epochs,
              verbose=2)

    if(fine_tune.lower() in ['true', '1', 't', 'y', 'yes']):
        base_model.trainable = True
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
                      loss =  'categorical_crossentropy' , metrics = ['accuracy'])

        history_object = model.fit(train_it,
                  validation_data=valid_it,
                  steps_per_epoch=train_it.samples/train_it.batch_size,
                  validation_steps=valid_it.samples/valid_it.batch_size,
                  epochs=my_epochs)
                  
    save_loss_plot(history_object.history, args)
    model.save(args.main_dir + '/' + h5modeloutput)


def save_loss_plot(history, args):
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Fruit Classification, Batch Size: ' + args.batch_size + ' Epochs: ' + args.epochs)
    plt.legend()
    plt.savefig(args.main_dir + '/' + 'model_b' + args.batch_size + '_e' + args.epochs + '.png')

if __name__ == "__main__":
    main()