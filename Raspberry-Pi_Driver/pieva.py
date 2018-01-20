import keras
import numpy as np
import random
from nn import neural_net
import h5py
import Usonic
import time

SAVED_MODEL = 'saved-models/ex1.h5'
NUM_INPUT = 7
NUM_SENSOR_RESOLUTION = 39
NN_SIZE = np.array([256, 256])
test_cnt = 0

model = neural_net(NUM_INPUT, NN_SIZE, SAVED_MODEL)
#model = neural_net(NUM_INPUT, NN_SIZE)
start = time.time()
while test_cnt < 3601:
    Dist = Usonic.distanceAll() / 10
    #In = np.random.random_integers(NUM_SENSOR_RESOLUTION, size=(NUM_INPUT))
    In = np.array([Dist[2],Dist[2],Dist[0],Dist[0],Dist[0],Dist[1],Dist[1]])
    In = In.astype(int)
    state = np.array([In])
    action = np.argmax(model.predict(state, batch_size = 1))
    test_cnt +=1
    #print(In)
    print("Number %d || action = %d || Dist = %d, %d, %d" %(test_cnt, action, Dist[0], Dist[1], Dist[2]))
end = time.time()
interval = end - start

print("Time Elaspsed: %d sec" %interval)

Usonic.GPIO.cleanup()
