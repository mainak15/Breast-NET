
from tensorflow.keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input,concatenate,add
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization,SeparableConv2D,Reshape,Permute,multiply
from keras.layers.convolutional import Conv2D, MaxPooling2D,AveragePooling2D
import time
import os.path
import itertools
#import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD ,RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
import math 
import keras
from tensorflow.keras.regularizers import l2
import os
import keras.backend as K
import tensorflow as TF
from sklearn.utils import class_weight
import numpy as np
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


train_data_dir = 'Dataset/train'
valid_data_dir = 'Dataset/val'
batch_size=256
checkpointer = ModelCheckpoint(
    filepath=os.path.join('Data', 'checkpoints', 'model_idc_4.hdf5'),
    verbose=1,
    save_best_only=True)


early_stopper = EarlyStopping(monitor='val_loss', patience=30)


tensorboard = TensorBoard(log_dir=os.path.join('Data', 'logs'))
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('Data', 'logs', 'model_idc_3' + '-' + 'training-' + \
        str(timestamp) + '.log'))

 
def get_generators():
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
                zoom_range=0.05,
                #brightness_range=(0.9, 1.1),
                #histogram_equalization=True,
                rotation_range=20,
                shear_range=0.05,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                #featurewise_center=True
                )

    val_datagen = ImageDataGenerator(rescale=1./255)

    

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(50, 50),
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb",
        classes=['0', '1'],
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(50, 50),
        batch_size=batch_size,
        color_mode="rgb",
        shuffle=False,
        classes=['0', '1'],
        class_mode='categorical')


    return train_generator, validation_generator

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

def _tensor_shape(tensor):
    return getattr(tensor, '_shape_val') if TF else getattr(tensor, '_keras_shape')

def CAM(input_tensor):
 
    init = input_tensor
    print(init.shape)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters =init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = SeparableConv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(se)
    se=BatchNormalization()(se)
    se = SeparableConv2D(filters, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = add([init, se])
    return x

def grouped_convolutions(x,
                     filters_1,
                     filters_2,
                     filters_3):
    
    conv_1 = SeparableConv2D(filters_1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_1_b = BatchNormalization()(conv_1)

    
    
    conv_2 = SeparableConv2D(filters_2, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_2_b = BatchNormalization()(conv_2)
    

    
    conv_3= SeparableConv2D(filters_3, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3_b = BatchNormalization()(conv_3)
    

    

    output = add([conv_1_b, conv_2_b, conv_3_b])
    print(output.shape[-1])
    
    return output

def Breast_NET_block(x, filters_1x1, filters_3x3, filters_5x5):

    x_0 = grouped_convolutions(x,
                         filters_1x1,
                         filters_3x3,
                         filters_5x5)


    x_2 = keras.layers.Activation("relu")(x_0)

    
    x_=CAM(x_2)

    return keras.layers.Activation("relu")(x_)

def srima():
    input_layer = Input(shape=(50, 50, 3))

    x = SeparableConv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='conv_1')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), padding='same',  name='max_pool_1_3x3')(x)

    x=Breast_NET_block(x,
                         filters_1x1=64,
                         filters_3x3=64,
                         filters_5x5=64)
    x=Breast_NET_block(x,
                         filters_1x1=96,
                         filters_3x3=96,
                         filters_5x5=96)
    x=Breast_NET_block(x,
                         filters_1x1=120,
                         filters_3x3=120,
                         filters_5x5=120)

    x1 = GlobalAveragePooling2D(name='avg_pool')(x)


    x1 = Dropout(0.4)(x1)
    x1 = Dense(2, activation='softmax', name='output')(x1)



    model = Model(input_layer, x1, name='Breast_NET')
    
   
    print(model.summary())
    #plot_model(model, to_file='model_BreaKHis_v10.png', show_shapes=True, show_layer_names=True)
    print("Number of layers in the base model: ", len(model.layers))

    return model

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

def train_model(model, nb_epoch, generators,callbacks=[]):

    train_generator, validation_generator = generators

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])    

    model._get_distribution_strategy = lambda: None
    
    class_weights = class_weight.compute_class_weight(
                class_weight ='balanced',
                classes = np.unique(train_generator.classes), 
                y = train_generator.classes)

    class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
    print(class_weights)

    classWeight = dict()
    his=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // batch_size,
        validation_data=validation_generator,
        validation_steps=len(validation_generator.filenames) // batch_size,
        epochs=nb_epoch,
        #class_weight=class_weights,
        #class_weight='balanced',
        callbacks=callbacks)
    
    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    training_loss = his.history['loss']
    validation_loss = his.history['val_loss']
    training_accuracy = his.history['accuracy']
    validation_accuracy = his.history['val_accuracy']
    epoch_count = range(1, len(training_loss) + 1)
    #N=num_epochs
    axs[0].plot(epoch_count, training_loss, 'r--')
    axs[0].plot(epoch_count, validation_loss, 'b-')
    axs[0].legend(['Training Loss', 'Validation Loss'])
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[1].plot(epoch_count, training_accuracy, 'r--')
    axs[1].plot(epoch_count, validation_accuracy, 'b-')
    axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    fig.savefig('model_idc_3.png')

    return model

def main(weights_file):

    
    model = srima()

    generators = get_generators()

    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    print('TensorFlow:', tf.__version__)

    generators = get_generators()

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())


    flops = graph_info.total_float_ops // 2
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    if weights_file is None:
        print("Loading network.")

        model = train_model(model, 300, generators,
                        [checkpointer, tensorboard, csv_logger, early_stopper])

    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
