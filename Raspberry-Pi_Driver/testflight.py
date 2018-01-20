import keras
import numpy as np
import random
from nn import neural_net
import h5py
import Usonic
import time
import drive
#from statistics import mean

SAVED_MODEL = 'saved-models/train7-100000.h5'
NUM_INPUT = 3
NUM_SENSOR_RESOLUTION = 39
NN_SIZE = np.array([256, 256])
test_cnt = 0

model = neural_net(NUM_INPUT, NN_SIZE, SAVED_MODEL)
#model = neural_net(NUM_INPUT, NN_SIZE)
start = time.time()
while test_cnt < 3601:
    Dist = Usonic.distanceAll()/2
    #In = np.random.random_integers(NUM_SENSOR_RESOLUTION, size=(NUM_INPUT))
    In = np.array([Dist[0],Dist[1],Dist[2]])
    In = In.astype(int)
    state = np.array([In])
    action = np.argmax(model.predict(state, batch_size = 1))
    
    #drive.GOFOWARD()
    if min(state[0]) > 50:
      drive.GOFOWARD()
    else:
      if action == 0:
        drive.GOLEFT()
      elif action == 1:
        drive.GORIGHT()
      elif action == 2:
        drive.GOFOWARD()
    test_cnt +=1
    #print(In)
    print("Number %d || action = %d || Dist = %d, %d, %d" %(test_cnt, action, Dist[0], Dist[1], Dist[2]))
end = time.time()
interval = end - start

print("Time Elaspsed: %d sec" %interval)

drive.GPIOEND()

