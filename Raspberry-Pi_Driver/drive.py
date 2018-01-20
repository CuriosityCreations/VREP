import RPi.GPIO as GPIO
import time

#GPIO Mode (Board/BCM)
GPIO.setmode(GPIO.BOARD)

#set GPIO Pins
GPIO_ENAB = (37, 38)
GPIO_MOTOR = (35, 36, 31, 33)

GPIO.setup(GPIO_ENAB[0], GPIO.OUT)
GPIO.setup(GPIO_ENAB[1], GPIO.OUT)
GPIO.setup(GPIO_MOTOR[0], GPIO.OUT)
GPIO.setup(GPIO_MOTOR[1], GPIO.OUT)
GPIO.setup(GPIO_MOTOR[2], GPIO.OUT)
GPIO.setup(GPIO_MOTOR[3], GPIO.OUT)

pwmA = GPIO.PWM(GPIO_ENAB[0], 120)
pwmB = GPIO.PWM(GPIO_ENAB[1], 120)
#pwmA.start(20)
#pwmB.start(20)

def GOFOWARD():
  pwmA.start(11)#50
  pwmB.start(13)
  GPIO.output(GPIO_MOTOR[0], GPIO.HIGH)
  GPIO.output(GPIO_MOTOR[1], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[2], GPIO.HIGH)
  GPIO.output(GPIO_MOTOR[3], GPIO.LOW)

def GOBACK():
  pwmA.start(14)
  pwmB.start(16)
  GPIO.output(GPIO_MOTOR[0], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[1], GPIO.HIGH)
  GPIO.output(GPIO_MOTOR[2], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[3], GPIO.HIGH)
  
def GORIGHT():
  pwmA.start(0)
  pwmB.start(0)
  GPIO.output(GPIO_MOTOR[0], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[1], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[2], GPIO.HIGH)
  GPIO.output(GPIO_MOTOR[3], GPIO.LOW)
  
def GOLEFT():
  pwmA.start(60)
  pwmB.start(60)
  GPIO.output(GPIO_MOTOR[0], GPIO.HIGH)
  GPIO.output(GPIO_MOTOR[1], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[2], GPIO.LOW)
  GPIO.output(GPIO_MOTOR[3], GPIO.LOW)

def GPIOEND():
  GPIO.cleanup()
  
if __name__ == "__main__":
  GOFOWARD()
  time.sleep(1)
  GOBACK()
  time.sleep(1)
  GORIGHT()
  time.sleep(1)
  GOLEFT()
  time.sleep(1)
  GPIOEND()
