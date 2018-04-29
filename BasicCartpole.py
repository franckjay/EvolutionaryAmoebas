#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:43:09 2018

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

nEpochs=1000
nAmoebas=10
LR=0.03
sigma=0.1
jitterBias=False
nActions,nStates=2,4

def playGame(env,model,nActions,nStates):
    state=env.reset()
    state = np.reshape(state, [1, nStates])
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
    dense=Dense(5,activation='relu')(dense)
    outLayer=Dense(nActions , activation='sigmoid')(dense)
    #outLayer=Dense(nActions , activation='softmax')(dense)
    return Model(inLayer, outLayer)

def getAmoebaDNA(model):
    DNA=[]
    biasDNA=[]
    #Grab first model in army as a baseline weight sample
    for l in range(1,len(model.layers)):
        weights=model.layers[l].get_weights()[0]#Grab weights for all layers
        bias=model.layers[l].get_weights()[1]
        DNA.append(weights) 
        biasDNA.append(bias)
    return DNA,biasDNA

def jitterDNA(models,sigma,nAmoebas):
    noise = []
    bias_noise = []
    for l in range(1,len(models[0].layers)):
        bias_shape = np.array(models[0].layers[l].get_weights()[1]).shape
        shape = np.array(models[0].layers[l].get_weights()[0]).shape

        #create a noise matrix to multiply the base weights by
        #uses polynomial distribution
        N = np.random.randn(nAmoebas, shape[0],shape[1])*sigma
        B = np.random.randn(nAmoebas, bias_shape[0])*sigma

        #add to containers
        noise.append(N)
        bias_noise.append(B)
    return noise,bias_noise

def addJitter(models,noise,bias_noise,baseDNA,baseBias,jitterBias):
    for w in range(nAmoebas):
        for l in range(1,len(models[0].layers)):
            newDNA = baseDNA[l-1] + noise[l-1][w]
            if jitterBias:
                new_bias = baseBias[l-1] + bias_noise[l-1][w]
            else:
                new_bias = baseBias[l-1]

            #set the weights on the current worker
            models[w].layers[l].set_weights((newDNA, new_bias))
    return models

def breedingGrounds(models,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias):
    for l in range(1,len(models[0].layers)):
        #print ("Noise: ",noise)
        weight_dot = noise[l-1]
        bias_dot = bias_noise[l-1]

        weight_dot = np.dot(weight_dot.transpose(1,2,0), A)
        #bias_dot = np.dot(bias_dot.transpose(1,2,0), A)

        baseDNA[l-1] += scale * weight_dot
        if jitterBias:
            baseBias[l-1] +=  scale* bias_dot
    return baseDNA,baseBias       

amoebaArmy = [buildModel(nActions,nStates) for i in range(nAmoebas)]
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
        print (i,"scores",np.mean(rewards) )
        #print (rewards)
    scale=LR/(sigma*nAmoebas)
    baseDNA,baseBias=breedingGrounds(amoebaArmy,noise,bias_noise,A,baseDNA,baseBias,scale,jitterBias)