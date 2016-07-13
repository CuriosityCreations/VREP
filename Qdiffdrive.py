import vrep
import trainvrep as tv
import numpy as np
import sys
import math
import keras
import random
from nn import neural_net, LossHistory
import time

#Initialize vrep and get the clientID
clientID = tv.Initialize()

#Car motor handlers
motorList = ('frontL', 'frontR', 'rearL', 'rearR')
motor_errorCode, motor_handles = tv.ObjectHandle(clientID, motorList)
fL_handle, fR_handle, rL_handle, rR_handle = motor_handles

#Collision handlers
#errorCode, collision_handle1=vrep.simxGetCollisionHandle(clientID,'Collision1',vrep.simx_opmode_blocking)
collision_handleList = ('Collision1', 'NULL')
collision_errorCode, handles = tv.CollisionHandle(clientID, collision_handleList)
collision_handle1 , NULL= handles

#Car_speed 
#LR-->LR (speed, > 0 go right, go foward = 0)
speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 20, 0, 0)

#Neural network parameters
NUM_INPUT = 3
ACT_OUTPUT = 3
nn_param = [256, 256]
observe = 1000  # Number of frames to observe before training.
epsilon = 1
final_epsilon = 0.001
train_frames = 100000  # Number of frames to play.
batchSize = 120
buffer = 50000

#Control parameters
target_speed = 10
fallback_sec = 1
nn_actPoint = 21

#Sarsa 0 parameters
PUNISH = -1000
GAMMA = 0.975
sarsa0P = ( PUNISH, GAMMA)

sensor_h = [] #empty array for sensor handles
sensor_val=np.array([]) #empty array for sensor measurements
sensor_state=np.array([]) #empty array for sensor measurements

#Initialize Neural Network
model = neural_net(NUM_INPUT, nn_param)


#Sensor handlers
sensorList = ('Proximity_sensor1', 'Proximity_sensor2', 'Proximity_sensor3')
sensor_errorCode, sensor_handles = tv.ObjectHandle(clientID, sensorList)

#Read sensor raw data (first time initial)
readval_errorCode, sensor_val, sensor_state = tv.INI_ReadProximitySensor(clientID, sensor_handles)

#for x in range(1,3+1):
    #errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Proximity_sensor'+str(x),vrep.simx_opmode_oneshot_wait)
    #sensor_h.append(sensor_handle) #keep list of handles
    #errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_streaming)                
    #sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
    #sensor_state=np.append(sensor_state,detectionState) #get list of values

train = np.floor(20*sensor_val).astype(int)
#Get New Rewards
#errorCode,linearVelocity,angularVelocity=vrep.simxGetObjectVelocity(clientID,cam_handle,vrep.simx_opmode_buffer)
#print(linearVelocity)
weightReadings = [1, 1, 1]
reward = np.dot(train,weightReadings)
reward = reward.astype(int)

state = np.array([train])
state = state.astype(int)

loss_log = []
replay = []
data_collect = []
maxroute = 0
driveCount = 0
trainCount = 0
errorCode, collisionState1=vrep.simxReadCollision(clientID,collision_handle1,vrep.simx_opmode_streaming)

while trainCount < train_frames:
  trainCount += 1
  print(trainCount)
  #Collision handler
  errorCode, collisionState1=vrep.simxReadCollision(clientID,collision_handle1,vrep.simx_opmode_buffer)

  print(collisionState1)
  print(maxroute)
  
  if collisionState1 == 1 or min(state[0]) <=1:
    if maxroute < driveCount and  trainCount > observe:
      maxroute = driveCount
      # Log the car's distance at this T.
      data_collect.append([trainCount, maxroute])
      
    driveCount = 0
    reward = PUNISH
    speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 8, -2, 1)
    time.sleep(fallback_sec)
    speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 8, 0, 0)
  
  if trainCount > observe:
    driveCount += 1
  
  # Choose an action.
  if random.random() < epsilon or trainCount < observe:
     action = np.random.randint(0, ACT_OUTPUT)  # random
  else:
    # Get Q values for each action.
    qval = model.predict(state, batch_size=1)
    action = (np.argmax(qval))  # best
  
  # Make an action
  if min(state[0]) > nn_actPoint:
    speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 8, 0, 0)
    action = 2
  else:
    if action == 0:
      speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 5, -7, 0)
    elif action == 1:
      speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 12, 7, 0)
    else:
      speed_errorCode = tv.MotorDifferential(clientID, motor_handles, 8, 0, 0)
  
  
  #Read sensor raw data
  readval_errorCode, sensor_val, sensor_state = tv.ReadProximitySensor(clientID, sensor_handles)
  #print(sensor_state)
  train = np.floor(20*sensor_val).astype(int)
  #print(train)
  
  new_state = np.array([train])
  new_state = new_state.astype(int)
  
  # Replay storage  lambda = 1 sarsa(0)
  replay.append((state, action, reward, new_state))
  
  #Get New Rewards
  #errorCode,linearVelocity,angularVelocity=vrep.simxGetObjectVelocity(clientID,cam_handle,vrep.simx_opmode_buffer)
  #print(linearVelocity)
  weightReadings = [1, 1, 1]
  reward = np.dot(train,weightReadings)
  reward = reward.astype(int)
  
  if trainCount > observe:
    # If we've stored enough in our buffer, pop the oldest.
    if len(replay) > buffer:
      replay.pop(0)
    
	# Randomly sample our experience replay memory
    minibatch = random.sample(replay, batchSize)
    
    # Get training values by Sarsa 0
    X_train, y_train = tv.sarsa0_minibatch(minibatch, model, sarsa0P)

    # Train the model on this batch.
    history = LossHistory()
    model.fit(
      X_train, y_train, batch_size=batchSize,
      nb_epoch=1, verbose=0, callbacks=[history]
    )
    loss_log.append(history.losses)

	
  state = new_state
  
  if epsilon > final_epsilon and trainCount > observe:
    epsilon -= (1/train_frames)
  print (epsilon)
  
  # Save the model every 25,000 frames.
  filename = 'train5'
  tv.save_models(filename, model, trainCount, 25000)
  
  #Write the CSV file
  tv.log_results(filename, data_collect, loss_log)
  

