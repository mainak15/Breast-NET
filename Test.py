
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,load_model
import os.path
import itertools
#import cv2
from glob import glob
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD
import re
import keras
import matplotlib.pyplot as plt
import numpy as np
import itertools

save_path = os.path.join('data', 'checkpoints', 'model_BreaKHis_v8.hdf5')
#valid_data_dir = 'data/test3'
valid_data_dir = 'Dataset1/test'
batch_size=8

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #plt.show()
    #plt.savefig('model_idcVGG16_confusion_matrix.png',bbox_inches = "tight")
    plt.close() 



def get_generators():
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator  = test_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(256, 256),
        batch_size=8,
        color_mode="rgb",
        shuffle=False,
        #classes=['0', '1'],
        #classes=['Grade_1', 'Grade_2', 'Grade_3'],
        classes=['Benign', 'Malignant'],
        #classes=['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma'],
        class_mode="categorical")


    return  test_generator


def predict_model(generators):
    test_generator = generators
    model = load_model(save_path)

    import tensorflow as tf
    from tensorflow.python.profiler.model_analyzer import profile
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
    print('TensorFlow:', tf.__version__)

    forward_pass = tf.function(
        model.call,
        input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())


    flops = graph_info.total_float_ops // 2
    print(f"FLOPS: {flops / 10 ** 9:.03} G")    
    print(model.summary())
    print("Number of layers in the base model: ", len(model.layers))
    print(test_generator.classes)
    predictions = model.predict_generator(
			generator = test_generator,
			workers=1,
			steps = len(test_generator.filenames) // batch_size+1 ,
			verbose = 1
			)
    pred=predictions
    predictions = np.argmax(predictions, axis=1)
    print(len(predictions))

    labels = (test_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predict = [labels[k] for k in predictions]
    filenames=test_generator.filenames
    print(len(filenames))
    print(len(predict))

          

    
    print('Confusion Matrix')
    cm=confusion_matrix(test_generator.classes, predictions)
    #target_names =['0', '1']
    #target_names =['Grade_1', 'Grade_2', 'Grade_3']
    target_names =['Benign', 'Malignant']
    #target_names=['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma']
    xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in target_names]


    plot_confusion_matrix(cm , 
                      normalize    = False,
                      target_names =target_names,
                      title        = "Confusion Matrix")
    print('Classification Report')
    #target_names = ['0', '1']
    target_names =['Benign', 'Malignant']
    #target_names =['Grade_1', 'Grade_2', 'Grade_3']
    #target_names=['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma']
    print(classification_report(test_generator.classes, predictions, target_names=target_names))
    # Plotting and estimation of FPR, TPR
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    plt.style.use('ggplot')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test_generator.classes, pred[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red'])
    #colors = cycle(['blue', 'red', 'green','orange','brown','pink', 'black','yellow'])    
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()


    return model

def main(weights_file):


    generators = get_generators()

    if weights_file is None:
        print("Loading saved model:")

        model = predict_model(generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    


if __name__ == '__main__':
    weights_file = None
    #print(len(data.classes))
    main(weights_file)
