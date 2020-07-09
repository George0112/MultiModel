# Containerized Tensorflow Inference Server Managed by Kubernetes

## Prerequrities

* [Multipass](https://snapcraft.io/install/multipass/ubuntu)

## Create Multipass VM

Launch a vm with sufficient resource
```multipass launch --name microk8s-vm --mem 8G --disk 40G --cpus 6```
Use a shell promt by running 
```multipass shell microk8s-vm```

## Install Microk8s in the VM
* We assume that this is complete in the vm shell.
```sudo snap install microk8s --classic --channel=1.17```
* Setup alias by 
```sudo snap alias microk8s.kubectl kubectl```
* Make k8s dir
```mkdir .kube/```
* Grant permission to user
```sudo usermod -a -G microk8s ubuntu```
**logout and login again**
* Store config to default position
```microk8s config > ~/.kube/config```
* Enable important add-on
```microk8s.enable dns dashboard storage```
* Check the status of k8s resources
```kubectl get all --all-namespaces```

Wait until the resources ready
Should get the similar result as 
```
ubuntu@microk8s-vm:~$ kubectl get all --all-namespaces
NAMESPACE     NAME                                                  READY   STATUS    RESTARTS   AGE
kube-system   pod/coredns-f7867546d-gf4bk                           1/1     Running   0          63m
kube-system   pod/heapster-v1.5.2-844b564688-btskj                  4/4     Running   0          62m
kube-system   pod/hostpath-provisioner-65cfd8595b-smjbw             1/1     Running   0          63m
kube-system   pod/kubernetes-dashboard-7d75c474bb-fnmjd             1/1     Running   0          63m
kube-system   pod/monitoring-influxdb-grafana-v4-6b6954958c-4vvq5   2/2     Running   0          63m


NAMESPACE     NAME                           TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                  AGE
default       service/kubernetes             ClusterIP   10.152.183.1     <none>        443/TCP                  67m
kube-system   service/heapster               ClusterIP   10.152.183.218   <none>        80/TCP                   63m
kube-system   service/kube-dns               ClusterIP   10.152.183.10    <none>        53/UDP,53/TCP,9153/TCP   63m
kube-system   service/kubernetes-dashboard   ClusterIP   10.152.183.167   <none>        443/TCP                  63m
kube-system   service/monitoring-grafana     ClusterIP   10.152.183.165   <none>        80/TCP                   63m
kube-system   service/monitoring-influxdb    ClusterIP   10.152.183.47    <none>        8083/TCP,8086/TCP        63m


NAMESPACE     NAME                                             READY   UP-TO-DATE   AVAILABLE   AGE
kube-system   deployment.apps/coredns                          1/1     1            1           63m
kube-system   deployment.apps/heapster-v1.5.2                  1/1     1            1           63m
kube-system   deployment.apps/hostpath-provisioner             1/1     1            1           63m
kube-system   deployment.apps/kubernetes-dashboard             1/1     1            1           63m
kube-system   deployment.apps/monitoring-influxdb-grafana-v4   1/1     1            1           63m

NAMESPACE     NAME                                                        DESIRED   CURRENT   READY   AGE
kube-system   replicaset.apps/coredns-f7867546d                           1         1         1       63m
kube-system   replicaset.apps/heapster-v1.5.2-6b794f77c8                  0         0         0       63m
kube-system   replicaset.apps/heapster-v1.5.2-6f5d55456                   0         0         0       62m
kube-system   replicaset.apps/heapster-v1.5.2-844b564688                  1         1         1       62m
kube-system   replicaset.apps/hostpath-provisioner-65cfd8595b             1         1         1       63m
kube-system   replicaset.apps/kubernetes-dashboard-7d75c474bb             1         1         1       63m
kube-system   replicaset.apps/monitoring-influxdb-grafana-v4-6b6954958c   1         1         1       63m
```
* Check the cluster info by
```kubectl cluster-info```
Should get similar result as
```
ubuntu@microk8s-vm:~$ kubectl cluster-info
Kubernetes master is running at https://127.0.0.1:16443
Heapster is running at https://127.0.0.1:16443/api/v1/namespaces/kube-system/services/heapster/proxy
CoreDNS is running at https://127.0.0.1:16443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
Grafana is running at https://127.0.0.1:16443/api/v1/namespaces/kube-system/services/monitoring-grafana/proxy
InfluxDB is running at https://127.0.0.1:16443/api/v1/namespaces/kube-system/services/monitoring-influxdb:http/proxy
```
* Forwarding port to k8s dashboard
```kubectl port-forward -n kube-system service/kubernetes-dashboard 10443:443 --address '0.0.0.0' &```
* Forwarding port to other service
```kubectl proxy --address='0.0.0.0' --accept-hosts='.*' &```
* Token of K8s dashboard
```
token=$(microk8s kubectl -n kube-system get secret | grep default-token | cut -d " " -f1)
microk8s kubectl -n kube-system describe secret $token
```

## Train Model (Skip if you want to use pretrained model)
If you want to use GPU, please refer to [TF GPU Support](https://www.tensorflow.org/install/gpu)
Currently, multipass doesn't support GPU passthrough
* Install necessary packages
```
sudo apt update
sudo apt install python3-pip unzip p-y
pip3 install --upgrade pip
pip3 install tensorflow-gpu pillow request
```
* Make a directory
```
mkdir catsdogs_tf_serve
cd catsdogs_tf_serve
```
* Download dataset
```
wget 140.114.79.72:9000/umc/sample.zip
unzip sample.zip
```
* Check python runtime is version 3
```
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

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
```

* Setup parameters
```
DATASET_PATH  = './sample'
IMAGE_SIZE    = (299, 299)
NUM_CLASSES   = 2
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 20
WEIGHTS_FINAL = 'model-inception_resnet_v2-final.h5'
```
* Import dataset
```
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
    

```

* Build Model Using Keras
```
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
```
* Train the Model
```
net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)
```

* Save the trained model
```
version = 1
export_path = os.path.join('./saved_model', str(version))
tf.keras.models.save_model(
    net_final,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

```
* Examine the saved model 
```saved_model_cli show --dir {export_path} --all```

## Tensorflow Serve

* Install Tensorflow Serve
```
sudo apt-get remove tensorflow-model-server
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install tensorflow-model-server -y
```
* Download pretrained model
```
wget 140.114.79.72:9000/umc/catsdogs_saved_model.zip
unzip catsdogs_saved_model.zip
```

* Start serving the model
```
tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=catsdogs \
  --model_base_path="/home/ubuntu/catsdogs_tf_serve/saved_model"
```
* Test the served model
```
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

img = image.load_img('./sample/valid/cats/cat.1001.jpg', target_size=(299,299))

x = image.img_to_array(img)
x = preprocess_input(x)
x = np.expand_dims(x, axis=0)

data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/catsdogs:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
class_names = ['cat', 'dog']

print("It's a %s" % class_names[np.argmax(predictions[0])])
```
## Build Docker Image

**Create a docker hub account if you hasn't**

* Install docker
```sudo apt install docker.io -y```

* Login to docker hub
```docker login```

* Edit Dockerfile
```
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM ubuntu:18.04

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL maintainer="gvasudevan@google.com"
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Expose ports
# gRPC
EXPOSE 8500

# REST
EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=catsdogs

# COPY Model
COPY saved_model /models/catsdogs

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server --port=8500 --rest_api_port=8501 \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
```
* Build docker image and push to dockerhub
```sudo docker build . -t <account>/catsdogs-tf-serve```
* Create kubernetes deployment
```kubectl create deployment catsdogs-serve --image=<account>/catsdogs-tf-serve```
* Check the deployment is running
```kubectl describe deployment catsdogs-serve```
* Forward port to pod
```kubectl port-forward deployment/catsdogs-serve 8501:8501 --address='0.0.0.0'```
* Use the python script above to test the inference server again