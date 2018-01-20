import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
import time
import drive

#pre_depth = 1

def quadcore_initialize():
    pool1 = ThreadPool(processes=1)
    pool2 = ThreadPool(processes=2)
    pool3 = ThreadPool(processes=3)
    pool4 = ThreadPool(processes=4)

    return pool1, pool2, pool3, pool4


def stereo_initialize(): 
    #left
    camera_matrixL = np.array([[219.44178, 0., 157.85633], [0., 219.82357, 122.50906], [0.,0.,1.]])
    dist_coefsL = np.array([-0.35878, 0.10911, -0.00069, -0.00088, 0.])

    #right
    camera_matrixR = np.array([[216.15875, 0., 163.36207], [0., 216.12017, 118.46570], [0.,0.,1.]])
    dist_coefsR = np.array([-0.37513, 0.13854, -0.00075, 0.00137, 0.])


    #extrinsic
    om = np.array([-0.00549, 0.01268, -0.06347])
    T = np.array([-63.36115, 0.61235, -1.71106])

    R1 = np.zeros(shape=(3,3))
    R2 = np.zeros(shape=(3,3))
    P1 = np.zeros(shape=(3,3))
    P2 = np.zeros(shape=(3,3))
    R = cv2.Rodrigues(om)

    cv2.stereoRectify(camera_matrixL, dist_coefsL, camera_matrixR, dist_coefsR,(320, 240), R[0], T, R1, R2, P1, P2, Q=None, alpha=-1, newImageSize=(0,0))
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrixL, dist_coefsL, R1, camera_matrixL, (320, 240), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camera_matrixR, dist_coefsR, R2, camera_matrixR, (320, 240), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y


def thread_stereomatch_quad(frameL,frameR,threadnumber,cut):
    frameL_buffer = frameL[(threadnumber-1)*60:threadnumber*60-cut, :]
    frameR_buffer = frameR[(threadnumber-1)*60:threadnumber*60-cut, :]

    window_size = 7
    min_disp = 16
    num_disp = 32-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        P1 = 8*3*window_size**2,
        P2 = 64*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 5,
        speckleWindowSize = 200,
        speckleRange = 2
    )
    disparity = stereo.compute(frameL_buffer, frameR_buffer).astype(np.float32) / 16.0
    disparity = cv2.normalize(disparity, (disparity-min_disp)/num_disp, 1, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #print(threadnumber)

    return disparity


if __name__ == '__main__':
  camera1 = cv2.VideoCapture(1) #left
  camera2 = cv2.VideoCapture(0) #right
  #resolution setting
  camera1.set(3,320)
  camera1.set(4,240)
  camera2.set(3,320)
  camera2.set(4,240)
  #quadcore initialize
  pool1, pool2, pool3, pool4 = quadcore_initialize()

  while 1:
      #camera initialization
      okL, frameL = camera1.read()
      okR, frameR = camera2.read()
      #stereo calib matrix map
      map1x, map1y, map2x, map2y = stereo_initialize()
      #camera calib
      frameL2 = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
      frameR2 = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)
      #gaussian blur
      frameL2 = cv2.GaussianBlur(frameL2,(5,5),20)
      frameR2 = cv2.GaussianBlur(frameR2,(5,5),20)
      #quadcore parallel stereo match
      async_T1 = pool1.apply_async(thread_stereomatch_quad, (frameL2, frameR2, 1, 0))
      async_T2 = pool2.apply_async(thread_stereomatch_quad, (frameL2, frameR2, 2, 0))
      async_T3 = pool3.apply_async(thread_stereomatch_quad, (frameL2, frameR2, 3, 0))
      async_T4 = pool4.apply_async(thread_stereomatch_quad, (frameL2, frameR2, 4, 0))
      #async get return data
      T1 = async_T1.get()
      T2 = async_T2.get()
      T3 = async_T3.get()
      T4 = async_T4.get()
      #disparity map combination
      disparity = np.vstack((T1,T2,T3,T4))
      disparity = cv2.dilate(disparity, None, iterations=3)
      #ROI distance (ahead)
      ROId = disparity[50:190, 145:175]
      mean = cv2.mean(ROId)
      depth = 43.2/mean[0]
      #Convert disparity map to 3 channels
      disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)
      #show results
      #cv2.imshow('frameL2',frameL2)
      #cv2.imshow('frameR2',frameR2)
      #cv2.imshow('disp',disparity)
      #cv2.imshow('ROI',ROId)

      k = cv2.waitKey(5) & 0xFF
      if k == 27:
          drive.GPIOEND()
          break
      if depth > 0.6:
          drive.GOFOWARD()
      elif depth < 0.6 and depth > 0.25:
          drive.GORIGHT()
      else:
          drive.GOBACK()
      #pre_depth = depth







