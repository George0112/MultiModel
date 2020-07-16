import os
import requests
import json
import numpy as np
import socket
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

class_names = ['cat', 'dog']

# def get_files(path):
#     files = [path]
#     files = [f for f in files if f.endswith('JPG') or f.endswith('jpg')]
#     if not len(files):
#         sys.exit('No images found by the given path!')

#     return files

index = 1
files = os.listdir('/home/chao/train')
HOST = '140.114.79.72'
PORT = 30800
# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((HOST, PORT))
times = []
for file in files:

    print(file)
    img = image.load_img('/home/chao/train/'+file, target_size=(299,299))

    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})\
    
    t1 = time.time()
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://140.114.79.72:30501/v1/models/catsdogs:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    t2 = time.time()

    times.append(t2-t1)

    clientMessage = '%s,%s' % (file.split('.')[0], class_names[np.argmax(predictions[0])])
    if index > 50:
        clientMessage = '%s,%s' % ('dog' if file.split('.')[0] == 'cat' else 'cat', class_names[np.argmax(predictions[0])])
        print("It's a %s, sending to concept drift detection" % ('dog' if class_names[np.argmax(predictions[0])] == 'cat' else 'cat'))
    else:
        print("It's a %s, sending to concept drift detection" % class_names[np.argmax(predictions[0])])
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    client.sendall(clientMessage.encode())
    # serverMessage = str(client.recv(1024), encoding='utf-8')
    # print('Server:', serverMessage)

    index += 1
    if index > 100:
        # client.close()
        break

print('Average time: %s' % str(np.average(times)))
