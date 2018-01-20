#HC SR04 ULTRA SONIC
#Fequency 40Hz
#Range 2~400cm

try:
    #Libraries
    import RPi.GPIO as GPIO 
    import time 
    import numpy as np

    #GPIO Mode (BOARD / BCM)
    GPIO.setmode(GPIO.BOARD)

    #set GPIO Pins
    GPIO_TRIGGER = (11, 12, 13)
    GPIO_ECHO = (15, 16, 18)

    GPIO.setup(GPIO_TRIGGER[0], GPIO.OUT)
    GPIO.setup(GPIO_TRIGGER[1], GPIO.OUT)
    GPIO.setup(GPIO_TRIGGER[2], GPIO.OUT)
    GPIO.setup(GPIO_ECHO[0], GPIO.IN)
    GPIO.setup(GPIO_ECHO[1], GPIO.IN)
    GPIO.setup(GPIO_ECHO[2], GPIO.IN)

    def distance():
        # set Trigger to HIGH
        GPIO.output(13, GPIO.HIGH)

        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(13, GPIO.LOW)

        Start = time.time()
        StopTime= time.time()
        StartTime = time.time()
        while GPIO.input(GPIO_ECHO[2]) == 0:
            StartTime = time.time()
            if StartTime - Start > 0.023:
              break

        while GPIO.input(GPIO_ECHO[2]) == 1:
            StopTime= time.time()
            if StopTime - StartTime > 0.023:
                break

        Distance = (StopTime - StartTime) *34300 / 2
        if Distance<2:
            Distance = 0
        elif Distance>399:
            Distance = 399
        return int(Distance)

    def distanceAll():
        # set Trigger to HIGH
        GPIO.output(GPIO_TRIGGER, GPIO.HIGH)

        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, GPIO.LOW)

        StartTime = time.time()
        StopTime= np.zeros(3)
        Distance = np.zeros(3)
        watchdog = time.time()

        # save StartTime
        while GPIO.input(GPIO_ECHO[0]) == 0 and GPIO.input(GPIO_ECHO[1]) == 0 and GPIO.input(GPIO_ECHO[2]) == 0 : pass
        StartTime = time.time()

        # save time of arrival
        while GPIO.input(GPIO_ECHO[0]) == 1 or GPIO.input(GPIO_ECHO[1]) == 1 or GPIO.input(GPIO_ECHO[2]) == 1:
            watchdog = time.time()
            if GPIO.input(GPIO_ECHO[0]) == 1:
                StopTime[0] = time.time()
            if GPIO.input(GPIO_ECHO[1]) == 1:
                StopTime[1] = time.time()
            if GPIO.input(GPIO_ECHO[2]) == 1:
                StopTime[2] = time.time()
            if watchdog - StartTime > 0.023:
                break

        # Calculate the distance in cm
        Distance = (StopTime - np.ones(3) * StartTime) *34300 / 2

        for numD in range(0, 3):
            if (Distance[numD] < 2):
                Distance[numD] = 0
            elif (Distance[numD] > 400):
                Distance[numD] = 399
    
        return Distance

    def distanceAll_resolution():

        # Resoluyion = 10 cm, total 40 intervals
        Dnormal = distanceAll() / 10

        return Dnormal.astype(int)

except KeyboardInterrupt: 
    print ("exit")
