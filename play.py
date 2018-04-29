#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:38:33 2018

@author: j
"""

import gym
import numpy as np
import keras
from keras.models import load_model
import cv2

gameName='SpaceInvaders-v0' 
env = gym.make(gameName)

#path="../Models/SpaceInvaders_model_INTERM.h5" #Mean score of 100 games: 179.55 +/- 114.070362058
#path="SpaceInvaders_model_FINAL_CPU.h5"#188.1 +/- 121.265782478

#path="SpaceInvaders_model_Evolution1000.h5"# 640.5 +/- 117.861147118
#path="SpaceInvaders_model_Evolution2000.h5"#667.15 +/- 150.438118507,best 905

path="SpaceInvaders_Conv_Evolution250.h5"#289.4 +/- 115.469216677, best 850
path="SpaceInvaders_Conv_Evolution400.h5"#path="SpaceInvaders_Conv_Evolution250.h5", best 730
path="SpaceInvaders_ConvPool_Evolution400.h5"#311.0 +/- 34.6121365998

path="SpaceInvaders_simple_Evolution2000.h5"

print path
amoebaOmega=load_model(path)

def preprocess(observation):
    #https://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

def play100(model,env):
    scores=[]
    maxScore=-999.
    for i in range(100):
        observation=env.reset()
        done = False
        tot_reward = 0.0
        if i==0:
            env.render()
        while not done:
            if i==0:
                env.render()
            observation=preprocess(observation)
            observation=np.expand_dims(observation,axis=0)
            Q = model.predict(observation)        
            action = np.argmax(Q)         
            observation, reward, done, info = env.step(action)
            tot_reward += reward
        if tot_reward>maxScore:
            maxScore=tot_reward
        scores.append(tot_reward)
    print ("Best score=",maxScore)
    print('Mean score of 100 games: {} +/- {}'.format(np.mean(scores),np.std(scores)))

play100(amoebaOmega,env)