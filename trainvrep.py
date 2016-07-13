import vrep
import numpy as np
import sys
import csv

def Initialize():
  # just in case, close all opened connections
  vrep.simxFinish(-1) 
  
  # connect to local host port 19999
  clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

  if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')

  else:
    print ('Connection not successful')
    sys.exit('Could not connect')

  return clientID

 
def ObjectHandle(clientID, objectList):
  # Get all the handles at once 
  shape = len(objectList)
  errorcode =[]
  handle = []
  
  for x in range(1,shape+1):
    get_errorCode, get_handle = vrep.simxGetObjectHandle(clientID, objectList[x-1], vrep.simx_opmode_oneshot_wait)
    errorcode.append(get_errorCode)
    handle.append(get_handle)

  return errorcode, handle
 

def CollisionHandle(clientID, objectList):
  # Get all the handles at once 
  shape = len(objectList)
  errorcode =[]
  handle = []
  
  for x in range(1,shape+1):
    get_errorCode, get_handle = vrep.simxGetCollisionHandle(clientID, objectList[x-1], vrep.simx_opmode_blocking)
    errorcode.append(get_errorCode)
    handle.append(get_handle)

  return errorcode, handle


def MotorDifferential(clientID, handleList, speed, diff, astern):
  # Get all the handles at once 
  shape = len(handleList)
  errorcode =[]
  
  if shape % 2 == 0:
    for x in range(1,shape+1,2):
      if astern :
        get_errorCode=vrep.simxSetJointTargetVelocity(clientID, handleList[x-1], -speed, vrep.simx_opmode_oneshot)
        errorcode.append(get_errorCode)
        get_errorCode=vrep.simxSetJointTargetVelocity(clientID, handleList[x], -(speed-diff), vrep.simx_opmode_oneshot)
        errorcode.append(get_errorCode)
      else:
        get_errorCode=vrep.simxSetJointTargetVelocity(clientID, handleList[x-1], speed, vrep.simx_opmode_oneshot)
        errorcode.append(get_errorCode)
        get_errorCode=vrep.simxSetJointTargetVelocity(clientID, handleList[x], speed-diff, vrep.simx_opmode_oneshot)
        errorcode.append(get_errorCode)
  else:
    sys.exit('Differential needs pairs of motors')

  return errorcode


def INI_ReadProximitySensor(clientID, sensor_handles):
  # Get all the handles at once 
  shape = len(sensor_handles)
  sensor_val=np.array([]) #empty array for sensor measurements
  sensor_state=np.array([]) #empty array for sensor measurements
  errorcode =[]
  
  for x in range(1,shape+1):
    get_errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_streaming)
    errorcode.append(get_errorCode)
    sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
    sensor_state=np.append(sensor_state,detectionState) #get list of values
  
  return  errorcode, sensor_val, sensor_state


def ReadProximitySensor(clientID, sensor_handles):
  # Get all the handles at once 
  shape = len(sensor_handles)
  sensor_val=np.array([]) #empty array for sensor measurements
  sensor_state=np.array([]) #empty array for sensor measurements
  errorcode =[]
  
  for x in range(1,shape+1):
    get_errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_buffer)
    errorcode.append(get_errorCode)
    sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
    sensor_state=np.append(sensor_state,detectionState) #get list of values
  
  return  errorcode, sensor_val, sensor_state


def sarsa0_minibatch(minibatch, model, sarsa0P):
  """This does the heavy lifting, aka, the training. It's super jacked."""
  X_train = []
  y_train = []
  # Loop through our batch and create arrays for X and y
  # so that we can fit our model at every step.
  for memory in minibatch:     
    # Get stored values.
    old_state_m, action_m, reward_m, new_state_m = memory
    #Get the number of input
    NUM_INPUT = len(old_state_m[0])
    # Get prediction on old state.
    old_qval = model.predict(old_state_m, batch_size=1)
    # Get prediction on new state.
    newQ = model.predict(new_state_m, batch_size=1)
    # Get our best move. (Sarsa 0 / Q-Learning)
    maxQ = np.max(newQ)
    #Get the number of output
    NUM_OUTPUT = len(newQ[0])

    y = np.zeros((1, NUM_OUTPUT))
    y[:] = old_qval[:]
        
    # Check for terminal state.
    if reward_m != sarsa0P[0]:  # non-terminal state
      update = (reward_m + (sarsa0P[1] * maxQ))
    else:  # terminal state
      update = reward_m

    # Update the value for the action we took.
    y[0][action_m] = update
        
    X_train.append(new_state_m.reshape(NUM_INPUT,))
    y_train.append(y.reshape(NUM_OUTPUT,))

  X_train = np.array(X_train)
  y_train = np.array(y_train)

  return X_train, y_train


def save_models(filename, model, trainCount, interval):
  if trainCount % interval == 0:
    model.save_weights('saved-models/' + filename + '-' +
                               str(trainCount) + '.h5',
                               overwrite=True)
    print("Saving model %s - %d" % (filename, trainCount))


def log_results(filename, data_collect, loss_log):
  # Save the results to a file so we can graph it later.
  with open('CSVresults/' + filename + '_route.csv', 'w') as data_dump:
    wr = csv.writer(data_dump)
    wr.writerows(data_collect)

  with open('CSVresults/' + filename + '_loss.csv', 'w') as lf:
    wr = csv.writer(lf)
    for loss_item in loss_log:
      wr.writerow(loss_item)
