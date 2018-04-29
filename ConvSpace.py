#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:15:48 2018

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

nEpochs=100
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
        #This should be fairly self-explanatory.
        action = model.predict(state)
        action = np.argmax(action)

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        state = new_state
        state=preprocess(state)
        state = np.expand_dims(state,axis=0)
    return total_reward


def buildConvModel(nActions,nStates,warmModel=False):
    inLayer=Input(shape=(84,84,1))
    conv=Conv2D(40,(2,2))(inLayer)
    conv=Conv2D(20,(4,4))(conv)
    conv=MaxPooling2D(pool_size(4,4))(conv)
    
    flat=Flatten()(conv)
    dense=Dense(48,activation='relu')(flat)
    dense=Dense(24,activation='relu')(dense)
    dense=Dense(12,activation='relu')(dense)
    outLayer=Dense(nActions , activation='softmax')(dense)
    model=Model(inLayer, outLayer)
    #If you already have a model, use it.
    if warmModel != False:
        model.load_weights(warmModel)
    return model


def getAmoebaDNA(model,bad_layers):
    DNA=[]
    biasDNA=[]
    good_layers={}
    goodLayer=0
    #Grab first model in army as a baseline weight sample
    for l in range(len(model.layers)):

        if l not in bad_layers:
            weights=model.layers[l].get_weights()[0]#Grab weights for all layers
            bias=model.layers[l].get_weights()[1]
            DNA.append(weights) 
            biasDNA.append(bias)
            good_layers[l]=goodLayer#Make sure this layer has weights!
            goodLayer+=1

    return DNA,biasDNA,good_layers

def jitterDNA(models,sigma,nAmoebas,bad_layers):
    noise = []
    bias_noise = []
    for l in range(len(models[0].layers)):
        if l not in bad_layers:
            bias_shape = np.array(models[0].layers[l].get_weights()[1]).shape
            shape = np.array(models[0].layers[l].get_weights()[0]).shape
            if len(shape) == 4:
                N = np.random.randn(nAmoebas, shape[0],shape[1],shape[2],shape[3])*sigma
            else:
                N = np.random.randn(nAmoebas, shape[0],shape[1])*sigma
            B = np.random.randn(nAmoebas, bias_shape[0])*sigma
            noise.append(N)
            bias_noise.append(B)
    return noise,bias_noise

def addJitter(models,noise,bias_noise,baseDNA,baseBias,jitterBias,good_layers):
    for w in range(nAmoebas):
        for l in range(len(models[0].layers)):
            if l in good_layers:
                newDNA = baseDNA[good_layers[l]] + noise[good_layers[l]][w]
                if jitterBias:
                    new_bias = baseBias[good_layers[l]] + bias_noise[good_layers[l]][w]
                else:
                    new_bias = baseBias[good_layers[l]]
                models[w].layers[l].set_weights((newDNA, new_bias))
    return models

def breedingGrounds(models,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias,good_layers):
    for l in range(len(models[0].layers)):
        if l in good_layers:
            weight_dot = noise[good_layers[l]]
            bias_dot = bias_noise[good_layers[l]]
            #If your weights are in a convolutional layer, they have a large shape!
            if len(weight_dot.shape)>4:
                weight_dot = np.dot(weight_dot.transpose(1,2,3,4,0), A)
            else:
                weight_dot = np.dot(weight_dot.transpose(1,2,0), A)
              
            baseDNA[good_layers[l]] += scale * weight_dot
            if jitterBias:
                baseBias[good_layers[l]] +=  scale* bias_dot
    return baseDNA,baseBias       

warmModel=False#3Layer Version.
bad_layers=[0,3,4]#Input layer,MaxPool,Flatten
amoebaArmy = [buildConvModel(nActions,nStates,warmModel) for i in range(nAmoebas)]


baseDNA,baseBias,good_layers=getAmoebaDNA(amoebaArmy[0],bad_layers)#Grab the first model as a benchmark
noise,bias_noise=jitterDNA(amoebaArmy,sigma,nAmoebas,bad_layers)

for i in range(nEpochs):
    rewards=np.zeros(nAmoebas)
    noise,bias_noise=jitterDNA(amoebaArmy,sigma,nAmoebas,bad_layers)
    amoebaArmy=addJitter(amoebaArmy,noise,bias_noise,baseDNA,baseBias,jitterBias,good_layers)
    
    for w in range(len(amoebaArmy)):
        rewards[w] = playGame(env,amoebaArmy[w],nActions,nStates)


    A=((rewards - np.mean(rewards)) / (np.std(rewards)))
    if i%10==0: 
        print (i,"scores",np.mean(rewards),np.std(rewards))
    scale=LR/(sigma*nAmoebas)
    baseDNA,baseBias=breedingGrounds(amoebaArmy,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias,good_layers)
    
amoebaArmy[0].save("SpaceInvaders_Conv_Evolution100.h5") 
