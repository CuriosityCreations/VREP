import vrep
import numpy as np
import sys
import math
import keras
import random
from nn import neural_net, LossHistory
import time
import csv

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    #sys.exit('Could not connect')

#Car handlers
errorCode, cam_handle = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_oneshot_wait)
errorCode, motor_handle=vrep.simxGetObjectHandle(clientID, 'motor_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, steer_handle=vrep.simxGetObjectHandle(clientID, 'steer_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, fl_handle=vrep.simxGetObjectHandle(clientID, 'fl_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, fr_handle=vrep.simxGetObjectHandle(clientID, 'fr_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, bl_handle=vrep.simxGetObjectHandle(clientID, 'bl_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, br_handle=vrep.simxGetObjectHandle(clientID, 'br_brake_joint#0',vrep.simx_opmode_oneshot_wait)
errorCode, collision_handle1=vrep.simxGetCollisionHandle(clientID,'Collision1',vrep.simx_opmode_blocking)
errorCode, collision_handle2=vrep.simxGetCollisionHandle(clientID,'Collision2',vrep.simx_opmode_blocking)
errorCode, collision_handle3=vrep.simxGetCollisionHandle(clientID,'Collision3',vrep.simx_opmode_blocking)
errorCode, collisionState1=vrep.simxReadCollision(clientID,collision_handle1,vrep.simx_opmode_streaming)
errorCode, collisionState2=vrep.simxReadCollision(clientID,collision_handle2,vrep.simx_opmode_streaming)
errorCode, collisionState3=vrep.simxReadCollision(clientID,collision_handle3,vrep.simx_opmode_streaming)
#errorCode,linearVelocity,angularVelocity=vrep.simxGetObjectVelocity(clientID,cam_handle,vrep.simx_opmode_streaming)

#Virtual Collision Sensor
#errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Proximity_sensor',vrep.simx_opmode_oneshot_wait)
#errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)

def log_results(filename, data_collect, loss_log):
  # Save the results to a file so we can graph it later.
  with open('CSVresults/' + filename + '_route.csv', 'w') as data_dump:
    wr = csv.writer(data_dump)
    wr.writerows(data_collect)

  with open('CSVresults/' + filename + '_loss.csv', 'w') as lf:
    wr = csv.writer(lf)
    for loss_item in loss_log:
      wr.writerow(loss_item)


def process_minibatch(minibatch, model):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        # Get prediction on new state.
        newQ = model.predict(new_state_m, batch_size=1)
        # Get our best move. I think?
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != PUNISH:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m

        # Update the value for the action we took.
        y[0][action_m] = update
        
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(3,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


#Control parameters
driveCount = 0
max_driveCount = 0
trainCount = 0
target_speed = 10
fallback_speed = -40
fallback_angle = -0.5
fallback_sec = 1

#Kinect Depth map
far = 3.3
near = 0.2

#Kinect sample point
kinectStart = 2
kinectInterval = 10
kinectHeight = 32
kinectMapsize = [48, 64]
resolution = 100
nn_actPoint = 90

#Neural network parameters
NUM_INPUT = 7
ACT_OUTPUT = 3
PUNISH = -1000
GAMMA = 0.975
nn_param = [256, 256]
observe = 1000  # Number of frames to observe before training.
epsilon = 1
final_epsilon = 0.001
train_frames = 200000  # Number of frames to play.
replay = []
loss_log = []
data_collect = []

params = {
  "batchSize": 200,
  "buffer": 50000,
  "nn": nn_param
}
batchSize = params['batchSize']
buffer = params['buffer']

model = neural_net(NUM_INPUT, nn_param)

# Initial the speed
errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,target_speed, vrep.simx_opmode_streaming)
errorCode, res, img = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_handle, vrep.simx_opmode_streaming)
im = np.array(img)
im.resize(kinectMapsize[0], kinectMapsize[1])  #Can be adjusted in VREP

#Depth 0~1 to distance 0.2 ~ 3.3
#dis = np.ones(kinectMapsize) * near + im * (far-near)
dis = im * resolution

#Get the readings
readings = []
for i in range(0, NUM_INPUT):
  readings.append(dis[kinectHeight][kinectStart + i * kinectInterval])
#errorCode,linearVelocity,angularVelocity=vrep.simxGetObjectVelocity(clientID,cam_handle,vrep.simx_opmode_buffer)
#print(linearVelocity)
weightReadings = [0.5, 0.7, 1, 2, 1, 0.7, 0.5]

#Get rewards
reward = np.dot(readings,weightReadings)
#print(reward)

state = np.array([readings])
state = state.astype(int)

maxroute = 0

while trainCount < train_frames:
  #Collision handler
  errorCode, collisionState1=vrep.simxReadCollision(clientID,collision_handle1,vrep.simx_opmode_buffer)
  errorCode, collisionState2=vrep.simxReadCollision(clientID,collision_handle2,vrep.simx_opmode_buffer)
  errorCode, collisionState3=vrep.simxReadCollision(clientID,collision_handle3,vrep.simx_opmode_buffer)
  if collisionState1 == 1 or collisionState2==1 or collisionState3==1:
    if maxroute < driveCount and  trainCount > observe:
      maxroute = driveCount
      # Log the car's distance at this T.
      data_collect.append([trainCount, maxroute])
      
      driveCount = 0
    reward = PUNISH
    errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,fallback_speed, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle, fallback_angle, vrep.simx_opmode_streaming)
    time.sleep(fallback_sec)
    errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,target_speed, vrep.simx_opmode_streaming)
    errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle, 0, vrep.simx_opmode_streaming)
  trainCount += 1
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
    errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,0,vrep.simx_opmode_streaming)
    action = 0
  else:
    if action == 0:
      errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle, 0.5, vrep.simx_opmode_streaming)
    elif action == 1:
      errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,-0.5,vrep.simx_opmode_streaming)
    else:
      errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,0,vrep.simx_opmode_streaming)

  
  #time.sleep(0.02)
  
  #Get next state
  errorCode, res, img = vrep.simxGetVisionSensorDepthBuffer(clientID, cam_handle, vrep.simx_opmode_buffer)
  im = np.array(img)
  im.resize(kinectMapsize[0], kinectMapsize[1])  #Can be adjusted in VREP
  
  #Depth 0~1 to distance 0.2 ~ 3.3
  #new_dis = np.ones(kinectMapsize) * near + im * (far-near)
  new_dis = im * resolution
  
  readings = []
  for i in range(0, NUM_INPUT):
    readings.append(new_dis[kinectHeight][kinectStart + i * kinectInterval])
  weightReadings = [0.5, 0.7, 1, 2, 1, 0.7, 0.5]

  new_state = np.array([readings])
  new_state = new_state.astype(int)
  
  # Replay storage  lambda = 1 sarsa(0)
  replay.append((state, action, reward, new_state))
  
  #Get New Rewards
  #errorCode,linearVelocity,angularVelocity=vrep.simxGetObjectVelocity(clientID,cam_handle,vrep.simx_opmode_buffer)
  #print(linearVelocity)
  reward = np.dot(readings,weightReadings)
  reward = reward.astype(int)
  
  #errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_buffer)
  #print(detectedPoint)
  
  #print("%f, %f, %f, %f, %f, %f, %f"%(dis[32][0], dis[32][10], dis[32][20], dis[32][30], dis[32][40], dis[32][50], dis[32][60]))
  print("%d, %d, %d, %d, %d, %d, %d"%(state[0][0], state[0][1], state[0][2], state[0][3], state[0][4], state[0][5], state[0][6]))
  print("%d, %d, %d, %d, %d, %d, %d"%(new_state[0][0], new_state[0][1], new_state[0][2], new_state[0][3], new_state[0][4], new_state[0][5], new_state[0][6]))
  print(action)
  print(reward)
  print (trainCount)
  print (maxroute)
  
  if trainCount > observe:
    # If we've stored enough in our buffer, pop the oldest.
    if len(replay) > buffer:
      replay.pop(0)
    
	# Randomly sample our experience replay memory
    minibatch = random.sample(replay, batchSize)
    
    # Get training values.
    X_train, y_train = process_minibatch(minibatch, model)

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
  filename = 'train1'
  if trainCount % 25000 == 0:
    model.save_weights('saved-models/' + filename + '-' +
                               str(trainCount) + '.h5',
                               overwrite=True)
    print("Saving model %s - %d" % (filename, trainCount))
  
  #Write the CSV file
  log_results(filename, data_collect, loss_log)
  
  #lock motor when velocity is zero
  #errorCode=vrep.simxSetJointTargetVelocity(clientID,motor_handle,1000, vrep.simx_opmode_streaming)
  #errorCode=vrep.simxSetJointForce(clientID,motor_handle,1, vrep.simx_opmode_oneshot)
  #errorCode=vrep.simxSetJointTargetPosition(clientID,steer_handle,0.1, vrep.simx_opmode_streaming)
  #errorCode=vrep.simxSetJointForce(clientID,fl_handle,100, vrep.simx_opmode_oneshot)
  
  
  
  


