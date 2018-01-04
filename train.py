
# coding: utf-8

import numpy as np
from collections import deque

import keras 
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from numpy import random
import pickle
import os,time
#import tensorflow as tf

DEBUG = False # whether to show if state changes unnormal
USE_CHROME = False # use the chrome browser to show playing, which is slow. 
ONLY_SHOW = False # Only play the best, not use random move

input_rows = 4 # here it is not a image, but the dimentions are alike
input_cols = 4
input_channels = 1
ACTIONS = 4 # actions: 4 directions
OBSERVATION = 1000 # how many observations before start to train
INITIAL_EPSILON = 0.5 # the prob of random trial. it will decrease until
FINAL_EPSILON = 0.0 if ONLY_SHOW else 0.0001 # ...until this number
EXPLORE = 8000 # how many steps from INITIAL_EPSILON to FINAL_EPSILON
REPLAY_MEMORY = 8000 # number of states to remember for replaying
BATCH = 200 # size of a minibatch
GAMMA = 0.99 # the decay rate

def buildmodel(show_model=False):
    main_input = Input(shape=(input_rows,input_cols,input_channels),
                       dtype='float32', name='main_input')
    x = Conv2D(64,(2,2),strides=(1,1),padding='same',activation='relu')(main_input)
    x = Conv2D(128,(2,2),strides=(1,1),padding='same',activation='relu')(x)
    conv_out = Flatten()(x)
    input2 = Input(shape=[1],dtype='float32',name='col_input')
    multi = keras.layers.concatenate([conv_out, input2])
    x = Dense(32)(multi)
    x = Dropout(0.5)(x)
    output = Dense(ACTIONS)(x)
    model = Model([main_input, input2], output)
    model.compile(loss='mse',optimizer='adam')
    if show_model: print(model.summary())
    return model


####### if you run at the first time:
if not os.path.exists('model.h5'):
    model = buildmodel(True)
    mem = deque(maxlen = REPLAY_MEMORY) # store the memories
    epsilon = INITIAL_EPSILON
    t=0
else:####### if you continue to run:
    with open('mem.pickle','rb') as f:
        (t,epsilon,mem)=pickle.load(f)
    from keras.models import load_model
    model = load_model('model.h5')
    print('lr0:',K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, 0.0001)
    print('lr1:',K.get_value(model.optimizer.lr))
    epsilon = INITIAL_EPSILON
##################################


if USE_CHROME: #initialize an API to the game
    from control import threes_API
    g = threes_API()
    time.sleep(0.5) # let the action to be finished
else:
    from control import Python_threes_API
    g = Python_threes_API() 

## log file
if not os.path.exists('log.csv'):
    with open('log.csv','a') as f:
        f.write('t,num,score,loss\n')


s_t = g.first_step()

valid_a = list(range(ACTIONS)) # valid actions
loss = -1 # a init value



while True: # start to loop
    if DEBUG: print('*********************************')
    if DEBUG and not USE_CHROME: input()#time.sleep(2)
    if DEBUG: print('t=%i,  epsilon=%f'%(t,epsilon),end='  ')
    if random.random()<=epsilon:
        if DEBUG: print('RANDOM MOVE!', end='  ')
        a_t = random.choice(valid_a)
    else:
        if DEBUG: print('Move by model.', end='  ')
        qs = model.predict(s_t)[0,valid_a]
        a_t = valid_a[np.argmax(qs)]
    if DEBUG: print(['UP','DOWN','LEFT','RIGHT'][a_t])
    # forward one step
    if DEBUG: print('Moving...',end=' ')
    s_t1, r_t, die = g.next_step(a_t)
    assert s_t1[0].max()<=1
    # if not change, this move is invalid
    if (s_t1[0]==s_t[0]).all(): 
        valid_a.remove(a_t)
        assert r_t<0 and s_t1[1][0,0]==s_t[1][0,0]
    else:
        valid_a = list(range(ACTIONS))
        assert die or r_t>=0 
    if DEBUG: print('Done.')
    # save it to memory
    if DEBUG: 
        print('NEW Memory: a_t=%i,r_t=%i,die=%i'%(a_t,r_t,die))
        print(s_t[0][0,:,:,0])
        
    if die:
        mem.append((s_t,a_t, r_t, None ,die))
        print('DEAD. n=%i max(n)=%i score=%i max(s)=%i'%(g.last_num,
              g.max_num, g.last_score, g.max_score))
        # the n is the coded value!!!!!!!!
    else:
        mem.append((s_t,a_t, r_t,s_t1,die))
        if DEBUG: print(s_t1[0][0,:,:,0])
    # update epsilon
    if epsilon > FINAL_EPSILON and t > OBSERVATION:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if epsilon < FINAL_EPSILON: epsilon = FINAL_EPSILON
    # trian the model if observations are enough
    if t > OBSERVATION:
        # sample a minibatch
        minibatch = random.choice(len(mem), min(BATCH,len(mem)))
#        # reinforce the bad punish
#        if r_t<0 and len(minibatch)>BATCH/3: # reinforce punish
#            minibatch[:int(BATCH/3)] = len(mem)-1 
        # initialize input and target
        input1 = np.zeros((BATCH, input_rows,input_cols,input_channels))
        input2 = np.zeros((BATCH,1))
        targets = np.zeros((BATCH, ACTIONS))
        # fill them
        for i,j in enumerate(minibatch):
            (s0_t,a0_t, r0_t, s0_t1, die0) = mem[j]
            input1[i:i+1] = s0_t[0]
            input2[i:i+1] = s0_t[1]
            targets[i] = model.predict(s0_t)
            if die0 or r0_t<0: 
                # if die or invalid, others can never be worse
                targets[i] = np.where(targets[i]<r0_t,r0_t+1,targets[i])
                targets[i,a0_t] = r0_t
            else:
                Qt1 = model.predict(s0_t1)
                targets[i,a0_t] = r0_t + GAMMA * np.max(Qt1)
        # train the model
        if DEBUG:print('Training the model...',end=' ')
        loss = model.train_on_batch([input1, input2],targets)
        if DEBUG:print('Done. loss=%f'%loss)
    # write log
    if die:
        with open('log.csv','a') as f:
            f.write('%i,%i,%i,%f\n'%(t,g.last_num,g.last_score,loss))
    # iteration
    s_t = s_t1
    t += 1
    # save the model every 10 times
    if (not DEBUG) and t%100==0: print('t=%i,  epsilon=%f, loss=%f'%(t,epsilon,loss))
    if t%500 ==0:
        print('saving model...',end=' ')
        model.save('model.h5')
        with open('mem.pickle','wb') as f:
            pickle.dump((t,epsilon,mem),f)
        print('Done.')


##### to check if the samples have any problem
def check_mem(mem, only_die=False):
    i = random.choice(len(mem))
    (s0_t,a0_t, r0_t, s0_t1, die0) = mem[i]
    while only_die and die0==False:
        i = random.choice(len(mem))
        (s0_t,a0_t, r0_t, s0_t1, die0) = mem[i]
    print(s0_t[0][0,:,:,0],s0_t[1])
    print(['UP','DOWN','LEFT','RIGHT'][a0_t])
    print('DEAD') if die0 else print(s0_t1[0][0,:,:,0],s0_t1[1])
    print('Reward:',r0_t)
check_mem(mem, False)

