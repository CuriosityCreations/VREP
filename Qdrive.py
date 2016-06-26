import vrep
import numpy as np
import sys
import math
import keras
import random
from nn import neural_net, LossHistory

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    #sys.exit('Could not connect')

errorCode, cam_handle = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_oneshot_wait)
errorCode, motor_handle=vrep.simxGetObjectHandle(clientID, 'motor_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, steer_handle=vrep.simxGetObjectHandle(clientID, 'steer_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, fl_handle=vrep.simxGetObjectHandle(clientID, 'fl_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, fr_handle=vrep.simxGetObjectHandle(clientID, 'fr_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, bl_handle=vrep.simxGetObjectHandle(clientID, 'bl_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, br_handle=vrep.simxGetObjectHandle(clientID, 'br_brake_joint#0',vrep.simx_opmode_oneshot_wait)

#Control parameters
driveCount = 0
max_driveCount = 0
trainCount = 0

#Kinect Depth map
far = 3.3
near = 0.2

#Kinect sample point
kinectStart = 0
kinectInterval = 10
kinectHeight = 30
kinectMapsize = [48, 64]

#Neural network parameters
NUM_INPUT = 3
nn_param = [256, 256]
observe = 1000  # Number of frames to observe before training.
epsilon = 1
train_frames = 1000000  # Number of frames to play.
params = {
  "batchSize": 400,
  "buffer": 50000,
  "nn": nn_param
}
batchSize = params['batchSize']
buffer = params['buffer']
model = neural_net(NUM_INPUT, nn_param)

# Initial the speed
errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,10000, vrep.simx_opmode_streaming)

while trainCount < train_frames:
  
  trainCount += 1
  driveCount += 1
  
  # Choose an action.
  if random.random() < epsilon or t < observe:
     action = np.random.randint(0, NUM_INPUT)  # random
  else:
    # Get Q values for each action.
    qval = model.predict(state, batch_size=1)
    action = (np.argmax(qval))  # best
  
  # Make action
  if action == 0:
    errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle, 0.5, vrep.simx_opmode_streaming)
  elif action == 1:
    errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,-0.5,vrep.simx_opmode_streaming)
  else:
    pass
  
  errorCode, res, img = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_handle, vrep.simx_opmode_oneshot_wait)
  im = np.array(img)
  im.resize(kinectMapsize[0], kinectMapsize[1])  #Can be adjusted in VREP
  
  #Depth to distance
  dis = np.ones(kinectMapsize) * near + im * (far-near)
  
  #Get the readings
  readings = []
  readings.append(dis[30][0])
  #Get rewards
  #rewards = 

  
  print("%f, %f, %f, %f, %f, %f, %f"%(dis[30][0], dis[30][10], dis[30][20], dis[30][30], dis[30][40], dis[30][50], dis[30][60]))

  #lock motor when velocity is zero
  #errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,1000, vrep.simx_opmode_streaming)
  #errorCode=vrep.simxSetJointForce(clientID,motor_handle,1, vrep.simx_opmode_oneshot)
  #errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,0.1, vrep.simx_opmode_streaming)
  #errorCode=vrep.simxSetJointForce(clientID,fl_handle,100, vrep.simx_opmode_oneshot)
  
  
  
  


