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