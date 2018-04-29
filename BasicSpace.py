#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:43:09 2018

@author: j
"""

import gym
import random
import numpy as np
from collections      import deque
import keras
import cv2
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model


def preprocess(observation):
    #https://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

def buildGame(gameName):
        env = gym.make(gameName)
        observation = env.reset() 
        observation=preprocess(observation)
        observation_size=np.shape(observation)
        action_size = env.action_space.n
        return env,observation_size,action_size

gameName='SpaceInvaders-v0'  #210,160,3 RGBs
env,nStates,nActions=buildGame(gameName)

print ("Number of actions: ",nActions,"number of states: ",nStates)

nEpochs=1000
nAmoebas=20
LR=0.03
sigma=0.1
jitterBias=False

def playGame(env,model,nActions,nStates):
    state=env.reset()
    state=preprocess(state)
    state = np.expand_dims(state,axis=0)
    total_reward=0.0
    done=False
    while not done:
        
        action = model.predict(state)

        #comment this out if you're not doing CartPole or something
        action = np.argmax(action)

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        #print (state)
        state = new_state
        #print (state)
        state=preprocess(state)

        state = np.expand_dims(state,axis=0)
        #print ("New State: ",state)
    return total_reward

def buildSingleModel(nActions,nStates,warmModel=False):

    inLayer=Input(shape=(84,84,1))
    flat=Flatten()(inLayer)
    dense=Dense(48,activation='relu')(flat)
    outLayer=Dense(nActions , activation='softmax')(dense)
    model=Model(inLayer, outLayer)
    if warmModel != False:
        model.load_weights(warmModel)
    #return Model(inLayer, outLayer)
    return model

def buildModel(nActions,nStates,warmModel=False):
#    print nStates
#    print nActions
    inLayer=Input(shape=(84,84,1))
    flat=Flatten()(inLayer)
    dense=Dense(48,activation='relu')(flat)
    dense=Dense(24,activation='relu')(dense)
    dense=Dense(12,activation='relu')(dense)
    #outLayer=Dense(nActions , activation='sigmoid')(dense)
    outLayer=Dense(nActions , activation='softmax')(dense)
    model=Model(inLayer, outLayer)
    if warmModel != False:
        model.load_weights(warmModel)
    #return Model(inLayer, outLayer)
    return model


def getAmoebaDNA(model):
    DNA=[]
    biasDNA=[]
    #Grab first model in army as a baseline weight sample
    for l in range(2,len(model.layers)):
        weights=model.layers[l].get_weights()[0]#Grab weights for all layers
        bias=model.layers[l].get_weights()[1]
        DNA.append(weights) 
        biasDNA.append(bias)
    return DNA,biasDNA

def jitterDNA(models,sigma,nAmoebas):
    noise = []
    bias_noise = []
    for l in range(2,len(models[0].layers)):
        bias_shape = np.array(models[0].layers[l].get_weights()[1]).shape
        shape = np.array(models[0].layers[l].get_weights()[0]).shape

        N = np.random.randn(nAmoebas, shape[0],shape[1])*sigma
        B = np.random.randn(nAmoebas, bias_shape[0])*sigma

        #add to containers
        noise.append(N)
        bias_noise.append(B)
    return noise,bias_noise

def addJitter(models,noise,bias_noise,baseDNA,baseBias,jitterBias):
    for w in range(nAmoebas):
        for l in range(2,len(models[0].layers)):
            newDNA = baseDNA[l-2] + noise[l-2][w]
            if jitterBias:
                new_bias = baseBias[l-2] + bias_noise[l-2][w]
            else:
                new_bias = baseBias[l-2]

            #set the weights on the current worker
            models[w].layers[l].set_weights((newDNA, new_bias))
    return models

def breedingGrounds(models,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias):
    for l in range(2,len(models[0].layers)):
        #print ("Noise: ",noise)
        weight_dot = noise[l-2]
        bias_dot = bias_noise[l-2]

        weight_dot = np.dot(weight_dot.transpose(1,2,0), A)
        #bias_dot = np.dot(bias_dot.transpose(1,2,0), A)

        baseDNA[l-2] += scale * weight_dot
        if jitterBias:
            baseBias[l-2] +=  scale* bias_dot
    return baseDNA,baseBias       

warmModel="SpaceInvaders_simple_Evolution2000.h5"#3Layer Version.
amoebaArmy = [buildSingleModel(nActions,nStates,warmModel) for i in range(nAmoebas)]

#warmModel="SpaceInvaders_model_Evolution2000.h5"#3Layer Version.
#amoebaArmy = [buildModel(nActions,nStates,warmModel) for i in range(nAmoebas)]
baseDNA,baseBias=getAmoebaDNA(amoebaArmy[0])#Grab the first model as a benchmark
noise,bias_noise=jitterDNA(amoebaArmy,sigma,nAmoebas)



for i in range(nEpochs):
    rewards=np.zeros(nAmoebas)
    noise,bias_noise=jitterDNA(amoebaArmy,sigma,nAmoebas)
    amoebaArmy=addJitter(amoebaArmy,noise,bias_noise,baseDNA,baseBias,jitterBias)
    
    for w in range(len(amoebaArmy)):
        rewards[w] = playGame(env,amoebaArmy[w],nActions,nStates)


    A=((rewards - np.mean(rewards)) / (np.std(rewards)))
    if i%10==0: 
        print (i,"scores",np.mean(rewards),np.std(rewards))
        #print (rewards)
    scale=LR/(sigma*nAmoebas)
    baseDNA,baseBias=breedingGrounds(amoebaArmy,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias)
    
amoebaArmy[0].save("SpaceInvaders_simple_Evolution2000.h5") 