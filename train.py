import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import itertools
import io
from lib import dataset
from lib.net import VCModel

epochs = 100
train_dataset, val_dataset = dataset.get_dataset()
model = VCModel()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def log_confusion_matrix(epoch, logs):
    test_pred = []
    test_label = []
    for image, label in val_dataset:
        test_pred_raw = model.predict(image)
        test_pred.extend(np.argmax(test_pred_raw, axis=1))
        test_label.extend(label)
    cm = sklearn.metrics.confusion_matrix(test_label, test_pred)
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    file_writer_cm = tf.summary.create_file_writer('logs' + '/cm')
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

def main():
  #keras.utils.plot_model(model, "model.png", show_shapes=True)
  model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  #model.build(input_shape=(None, 8, 224, 224, 3))
  #model.summary()
  #exit(1)

  callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='model/VCModel',
        save_best_only=True,
        monitor='val_accuracy',
        mode='auto',
        verbose=1,
    ),
    keras.callbacks.TensorBoard(
        log_dir='logs',
    ),
    #keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
  ]

  history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
  )
  print (history.history)



if __name__ == '__main__':
  main()
