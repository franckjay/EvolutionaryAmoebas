#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:44:14 2018

@author: j
"""

import gym
import keras
import numpy as np
#https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
#https://gist.github.com/MaxwellRebo/7d4a9924e721870d6bc9554d727b0c31

from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.optimizers import Adam
from keras.models import Sequential, Model

game = 'CartPole-v1'
env = gym.make(game)

nEpochs=200
nAmoebas=64
LR=0.03
sigma=0.1

nActions,nStates=2,4
print ("This one works, but is not learning anything.")
def playGame(env,model,nActions,nStates):
    state=env.reset()
    #print (state)
    state = np.reshape(state, [1, nStates])
    #print (state)
    FRAME_COUNT=0
    total_reward=0.0
    done=False
    while not done:
        action = model.predict(state)[0]
        #print (action)

        #comment this out if you're not doing CartPole or something
        action = np.argmax(action)

        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        #print (state)
        state = new_state
        #print (state)
        state = np.reshape(state, [1, nStates])
        #print ("New State: ",state)
    return total_reward

def buildModel(nActions,nStates):
    inLayer=Input(shape=(nStates,))
    dense=Dense(24,activation='relu')(inLayer)
    dense=Dense(24,activation='relu')(dense)
    #outLayer=Dense(nActions , activation='sigmoid')(dense)
    outLayer=Dense(nActions , activation='softmax')(dense)
    return Model(inLayer, outLayer)

amoebaArmy = [buildModel(nActions,nStates) for i in range(nAmoebas)]

def breedingGrounds(models,sigma,A,scaler):
    jitterBiases=False
    #We are just adding jitter here
    if not scaler:
        for i in range(len(models)):
            for l in range(1,len(models[i].layers)):
                weights=models[i].layers[l].get_weights()#Grab weights for all layers
                #print (weights)
                shape=np.array(weights[0]).shape
                weights[0]+=np.random.randn(int(shape[0]),int(shape[1]))#Adds jitter to weight
                if jitterBiases:
                    shape=np.array(weights[1]).shape
                    weights[1]+=np.random.randn(int(shape[0]))*sigma# First array is weights, 2nd is bias 
                models[i].layers[l].set_weights(weights)#Update model weights
                #print ("Adjusted: ", models[i].layers[l].get_weights())
    #Based on the rewards for each amoeba, we will move the weights around            
    else:
        for i in range(len(models)):
            for l in range(1,len(models[i].layers)):
                weights=models[i].layers[l].get_weights()#Grab weights for all layers
                weights[0]+=scaler*np.dot(weights[0],A[w])
                if jitterBiases:
                    weights[1]+=scaler*np.dot(weights[1],A[w]) 
                models[i].layers[l].set_weights(weights)#Update model weights        
    return models



for i in range(nEpochs):
    rewards=np.zeros(nAmoebas)
    #print ("Insert jitter to layers ")
    amoebaArmy=breedingGrounds(amoebaArmy,sigma,False,False)
    
    for w in range(len(amoebaArmy)):
        rewards[w] = playGame(env,amoebaArmy[w],nActions,nStates)

	#get standard deviation of each reward
	#add a small amount to account for divide by 0
    A=((rewards - np.mean(rewards)) / (np.std(rewards)))
    if i%10==0: 
        print (i,"scores",np.mean(rewards) )
        #print (rewards)
    scaler=LR/(sigma*nAmoebas)
    amoebaArmy=breedingGrounds(amoebaArmy,sigma,A,scaler)