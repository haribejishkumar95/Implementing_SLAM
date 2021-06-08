# -*- coding: utf-8 -*-
'''
Author : Harikrishnan Bejishkumar
Course: Graduate Diploma in Mechatronics
Institute: Wellington Institute of Technology

This program is to create a simulation environment for the AI robot to work in. This is made using
"KIVY"; an open source Python Integrated library for rapid development of applications that make
use of innovative user interfaces. This program has imported brain of the robot to incorporate those behaviour
into the widget made using kivy.
'''
import numpy as np #Importing the numpy library
import csv
import matplotlib.pyplot as plt
import time
import mpu6050 as mpu #library for mpu6050 module
import RPi.GPIO as GPIO

dt = 0.001 #sampling time from mpu6050 datasheet
x_init = 0
y_init = 0
wz_init = 0
GPIO.setmode(GPIO.BCM)

#Pins for servo motors
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

#Pins for Ultrasonic sensors
GPIO.setup(23, GPIO.IN)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(25, GPIO.IN)
GPIO.setup(16, GPIO.OUT)

#Pins for DC motors
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(5, GPIO.OUT)

#Pins for Gyroscope
GPIO.setup(2, GPIO.OUT)
GPIO.setup(3, GPIO.IN)


from Brain import Dqn #Importing AI Deep Q learning Reinforcement algorithm






brain = Dqn(6,4,0.9) #brain of the robot; initialized globally with the imported Dqn class from ai code
action2rotation = [0,30,-30, 60] #To select a rotaion depending on the value of action[0(No rotation),1(rotate 20deg clockwise),2(rotate 20deg anti-clockwise)]
last_reward = 0 #to record the reward it gets for certain action
scores = [] #record the score depending on the reward

Initial_update = True #To ensure switch is pressed


           
orientation =  # direction of the car 
acceleration = #speed of the car
last_signal = [sensor1, sensor2, sensor3, sensor4, orientation, acceleration] # batch of inputs which includes 4 signals from the sensor and orientation and -orientation
action = brain.update(last_reward, last_signal) # playing the action using the dqn model
scores.append(brain.score()) # appending the mean of the last 100 rewards to the reward window
rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)

fwall; #distance of the frontwall from the centre of the robot (distance from sensor + 7cm)
rwall; #distance of the rightwall from the centre of the robot (distance from sensor + 7cm)
lwall; #distance of the leftwall from the centre of the robot (distance from sensor + 7cm)
bwall;#distance of the backwall from the centre of the robot (distance from sensor + 7cm)
   
   
if fwall > 0: # if the car is on the wall
    #slowed down
    last_reward = -10# and reward = -10
else: # otherwise      
    # maintain the normal speed
    last_reward = -0.2 # and it gets bad reward (-0.2), which is a living penalty
    if distance < last_distance: # however if it getting close to the goal
        last_reward = 0.1 # it still gets slightly positive reward 0.1
   
   
#if if reached the goal        
if int(distance) < 6:
    self.car.velocity = Vector(3, 0).rotate(self.car.angle)#it is stopped
    print("Winner")
    last_reward = 1#gets the highest reward
    Initial_update = True#makes sure, nothing gets updated after that unless start is pressed again
    plt.show()#plot the graph once finished the current training
   
   
    #return self.add_widget(self.l)#print the label on the canvas; "WINNER!"
 
last_distance = distance#updating the last distance to the current
'''            
code for storing the data of the training : The dataset of the training .csv file contains nothing but the
set of all (x,y) point of its movement through the screen. Each time switch is pressed, a new training starts and
a new .csv file  is created to store the data
'''
nl = '\n'
if last_graph != graph: #if start is pressed again
    i = graph#create a new .csv file
   
   
graph_data = 'data{}.csv'.format(i)
with open(graph_data, 'a') as file:
           
    file.write(f"{self.car.x} , {self.car.y}{nl}")#appending the position of the car each time it takes a step into .csv file
                           
    x_values, y_values = np.loadtxt(graph_data, dtype='int,int' ,delimiter=',', usecols=(0, 1), unpack = True)#reading the value
                           
                               
    plt.plot(x_values, y_values)
                                   
    plt.title('Training data of the car')
                               
    plt.xlabel('Movement along the width')
    plt.ylabel('Movement along the height')
           
       
last_graph = graph#storing the last number of START switch pressed
class KalmanFilter(object):
    def __init__(self, a_x,a_y, std_acc, x_std_meas, y_std_meas):
        """
         dt: sampling time 
         a_x: acceleration in x-direction
         a_y: acceleration in y-direction
         std_acc: process noise magnitude
         x_std_meas: standard deviation of the measurement in x-direction
         y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        
        # Define the  control input variables
        self.u = np.matrix([[a_x],[a_y]])
        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.matrix([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt,0],
                            [0,dt]])
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        #Initial Process Noise Covariance
        self.Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])
        #Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])
        
    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # Update time state
        #x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]
        
    def update(self,z):
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  #Eq.(11)
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))   #Eq.(12)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P   #Eq.(13)
        return self.x[0:2]
        
T = KalmanFilter(1,1,1,1,0.1,0.1)
print(T.x)
z = 0
while(True):
    z +=1
    
    T.predict()
    T.update(z)#updated z is given to update function at each time
    print(T.x)
    time.sleep(1)

