import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import os

print('TensorFlow version: {}'.format(tf.__version__))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
DATASET_PATH  = './sample'
IMAGE_SIZE    = (299, 299)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 1
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/valid',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)
# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.optimizers import Adam
net = InceptionResNetV2(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)
)
x = net.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(0.5)(x)
output_layer = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
net_final = Model(inputs=net.input, outputs=output_layer)

for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(net_final.summary())

net_final.fit(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)
                        
version = 1
export_path = os.path.join('/models/catsdogs', str(version))
tf.keras.models.save_model(
    net_final,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import model_management_pb2
from tensorflow_serving.config import model_server_config_pb2

import grpc

def add_model_config(host, name, base_path, model_platform):
  channel = grpc.insecure_channel(host) 
  stub = model_service_pb2_grpc.ModelServiceStub(channel)
  request = model_management_pb2.ReloadConfigRequest() 
  model_server_config = model_server_config_pb2.ModelServerConfig()

  #Create a config to add to the list of served models
  config_list = model_server_config_pb2.ModelConfigList()       
  one_config = config_list.config.add()
  one_config.name= name
  one_config.base_path=base_path
  one_config.model_platform=model_platform

  model_server_config.model_config_list.CopyFrom(config_list)

  request.config.CopyFrom(model_server_config)

  print(request.IsInitialized())
  print(request.ListFields())

  response = stub.HandleReloadConfigRequest(request,10)
  if response.status.error_code == 0:
      print("Reload sucessfully")
  else:
      print("Reload failed!")
      print(response.status.error_code)
      print(response.status.error_message)


add_model_config(host="140.114.79.72:30500", 
                    name="catsdogs", 
                    base_path="/models/catsdogs", 
                    model_platform="tensorflow")