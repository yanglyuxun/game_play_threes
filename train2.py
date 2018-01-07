
# coding: utf-8
'''
The new train method
'''

import numpy as np
from collections import deque

import keras 
from keras import backend as K
from keras.models import Model
from keras.initializers import TruncatedNormal
from keras.layers import Input, Embedding, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from numpy import random
import pickle
import os,time
#import tensorflow as tf

DEBUG = False # whether to show if state changes unnormal
USE_CHROME = False # use the chrome browser to show playing, which is slow. 
ONLY_SHOW = False # Only play the best, not use random move

ACTIONS = 4 # actions: 4 directions
OBSERVATION = 1000 # how many observations before start to train
INITIAL_EPSILON = 0.5 # the prob of random trial. it will decrease until
FINAL_EPSILON = 0.0 if ONLY_SHOW else 0.01 # ...until this number
EXPLORE = 5000 # how many episodes from INITIAL_EPSILON to FINAL_EPSILON
REPLAY_MEMORY = 100000 # number of states to remember for replaying
BATCH = 1000 # size of a minibatch
GAMMA = 0.99 # the decay rate

## TODO: is conv appropriate??? Try all Dense!!
def buildmodel(show_model=False):
    input1 = Input(shape=(4,4,1),dtype='float32',name='ma')
    input2 = Input(shape=(4,4,2),dtype='float32',name='mb')
#    x = Conv2D(64,(2,2),strides=(1,1),padding='same',
#               activation='relu',
#               kernel_initializer=TruncatedNormal(),
#               bias_initializer='zeros')(input1)
#    x = Conv2D(128,(2,2),strides=(1,1),padding='same',
#               activation='relu',
#               kernel_initializer=TruncatedNormal(),
#               bias_initializer='zeros')(x)
    flat1 = Flatten()(input1)
    flat2 = Flatten()(input2)
    input3 = Input(shape=[1],dtype='float32',name='mc')
    x = keras.layers.concatenate([flat1,flat2,input3])
    x = Dense(128,kernel_initializer=TruncatedNormal(),
              bias_initializer='zeros',activation='relu')(x)
    #x = Dropout(0.5)(x) large data set, no need for dropout
    x = Dense(64,kernel_initializer=TruncatedNormal(),
              bias_initializer='zeros',activation='relu')(x)
    x = Dense(64,kernel_initializer=TruncatedNormal(),
              bias_initializer='zeros')(x)
    x = Dense(ACTIONS,kernel_initializer=TruncatedNormal(),
                   bias_initializer='zeros')(x)
    model = Model([input1,input2,input3], x)
    model.compile(loss='mse',optimizer='adam')
    if show_model: print(model.summary())
    return model


####### if you run at the first time:
if not os.path.exists('model.h5'):
    model = buildmodel(True)
    mem = deque(maxlen = REPLAY_MEMORY) # store the memories
    records = deque(maxlen = REPLAY_MEMORY) # only record numbers
    epsilon = INITIAL_EPSILON
    t=0
else:####### if you continue to run:
    with open('mem.pickle','rb') as f:
        (t,epsilon,mem,records)=pickle.load(f)
    from keras.models import load_model
    model = load_model('model.h5')
    print('lr0:',K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, 0.0001)
    print('lr1:',K.get_value(model.optimizer.lr))
    epsilon = INITIAL_EPSILON
##################################


# use the game iteration directly
from game.python_threes import play_game
 

## log file
if not os.path.exists('log.csv'):
    with open('log.csv','a') as f:
        f.write('t,num,score,loss\n')

####### loop part
def special_norm(m):
    ''' a special way to normalize m to:
        {0}: empty place (unchanged)
        Then change 1 to 2, because 1 and 2 can be combined to 3
        The problem is, 1 and 1, 2 and 2 cannot be combined,
        so additional layers are needed
        [1,2] : all the values are normalized to [1,2]
        output: 4*4*1
        '''
    m1 = np.where(m==1,2, m) # change all 1 to 2
    mmax = m1.max()
    mmin = np.where(m1==0,mmax,m1).min() # min nonzero value
    d = mmax - mmin
    return np.where(m1==0,0, (m1-mmin)/d+1.).reshape((4,4,1))

def gene_comb_layer(m):
    '''from m, generate the layers represent 
    whether a value can be combined with the neighber
    output 4*4*2'''
    mv, mh = np.zeros((4,4)),np.zeros((4,4))
    for i in range(3):
        for j in range(4):
            if (m[i,j]>2 and m[i,j]==m[i+1,j]) or m[i,j]+m[i+1,j]==3:
                mv[i,j] = 1
            if (m[j,i]>2 and m[j,i]==m[j,i+1]) or m[j,i]+m[j,i+1]==3:
                mh[j,i] = 1
    return np.stack((mv,mh),axis=-1)

def get_input(m,tile):
    '''get the input for the NN'''
    ma = np.expand_dims(special_norm(m), 0)
    mb = np.expand_dims(gene_comb_layer(m), 0)
    mc = np.array([[tile]])
    return [ma,mb,mc]

def to_score(m):
    '''cal the score of m'''
    return np.where(m < 3, 0, 3.0**(m-2)).astype(np.int).sum()
    
loss = -1 # a init value
while True: # start to loop
    if DEBUG: print('*********************************')
    if DEBUG and not USE_CHROME: input()#time.sleep(2)
    if DEBUG: print('t=%i,  epsilon=%f'%(t,epsilon),end='  ')
    
    
    #### do a whole episode
    episode = []
    g = play_game()
    
    m, tile, valid = g.send(None)#the first state
    if tile>3: tile=3 # when next>=3, we don't know it
    inp = get_input(m,tile) # the input for NN
    while valid:
        if random.random()<=epsilon:
            a_t = random.choice(valid)
        else:
            qs = model.predict(inp)[0,valid]
            a_t = valid[np.argmax(qs)]
        #if DEBUG: print(['UP','DOWN','LEFT','RIGHT'][a_t])
        # forward one step
        #if DEBUG: print('Moving...',end=' ')
        m1, tile1, valid1 = g.send(a_t)
        if tile1>3: tile1=3
        inp1 = get_input(m1,tile1)
        episode.append((m,tile,valid,inp,a_t,
                        m1,tile1,valid1,inp1,
                        to_score(m)))
        # update
        m,tile,valid,inp = m1,tile1,valid1,inp1
    g.close()
    score = to_score(m) # the final score when dead
    for (m,tile,valid,inp,a_t,m1,tile1,valid1,inp1,score0) in episode:
        mem.append((valid,inp,a_t,valid1,inp1,score-score0))
        records.append((m,tile,m1,tile1,score0))
    # write log
    if t%100==0:
        with open('log.csv','a') as f:
            f.write('%i,%i,%i,%f\n'%(t,m.max(),score,loss))
## TODO: continue    
    
    # save it to memory
#    if DEBUG: 
#        print('NEW Memory: a_t=%i,r_t=%i,die=%i'%(a_t,r_t,die))
#        print(s_t[0][0,:,:,0])
#        
#    if die:
#        mem.append((s_t,a_t, r_t, None ,die))
#        print('DEAD. n=%i max(n)=%i score=%i max(s)=%i'%(g.last_num,
#              g.max_num, g.last_score, g.max_score))
#        # the n is the coded value!!!!!!!!
#    else:
#        mem.append((s_t,a_t, r_t,s_t1,die))
#        if DEBUG: print(s_t1[0][0,:,:,0])
    # update epsilon
    if epsilon > FINAL_EPSILON and t > OBSERVATION:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if epsilon < FINAL_EPSILON: epsilon = FINAL_EPSILON
    # trian the model if observations are enough
    if t > OBSERVATION:
        first_train = True
        while first_train or loss>20000: # control the loss
            # sample a minibatch
            minibatch = random.choice(len(mem), min(BATCH,len(mem)))
    #        # reinforce the bad punish
    #        if r_t<0 and len(minibatch)>BATCH/3: # reinforce punish
    #            minibatch[:int(BATCH/3)] = len(mem)-1 
            # initialize input and target
            input1 = np.zeros((BATCH, 4,4,1))
            input2 = np.zeros((BATCH, 4,4,2))
            input3 = np.zeros((BATCH, 1))
            targets = np.zeros((BATCH, ACTIONS))
            # fill them
            for i,j in enumerate(minibatch):
                (valid,inp,a_t,valid1,inp1,r) = mem[j]
                input1[i:i+1] = inp[0]
                input2[i:i+1] = inp[1]
                input3[i:i+1] = inp[2]
                targets[i] = model.predict(inp)
                if valid: 
                    Qt1 = model.predict(inp1)
                    targets[i,a_t] = r + GAMMA * np.max(Qt1)
                else:
                    targets[i,a_t] = r
            # train the model
            if not first_train: print('Training the model...',end=' ')
            loss = model.train_on_batch([input1,input2,input3],targets)
            if not first_train: print('Done. loss=%f'%loss)
            first_train = False
    # iteration
    t += 1
    # save the model every 10 times
    if (not DEBUG) and t%50==0: print('t=%i,  epsilon=%f, loss=%f'%(t,epsilon,loss))
    if t%50 ==0:
        print('saving model...',end=' ')
        model.save('model.h5')
        with open('mem.pickle','wb') as f:
            pickle.dump((t,epsilon,mem,records),f)
        print('Done.')


##### to check if the samples have any problem
#def check_mem(mem, only_die=False):
#    i = random.choice(len(mem))
#    (s0_t,a0_t, r0_t, s0_t1, die0) = mem[i]
#    while only_die and die0==False:
#        i = random.choice(len(mem))
#        (s0_t,a0_t, r0_t, s0_t1, die0) = mem[i]
#    print(s0_t[0][0,:,:,0],s0_t[1])
#    print(['UP','DOWN','LEFT','RIGHT'][a0_t])
#    print('DEAD') if die0 else print(s0_t1[0][0,:,:,0],s0_t1[1])
#    print('Reward:',r0_t)
#check_mem(mem, False)

######## random moving experiment
scores=[]
nums = []
for _ in range(2644):
    episode = []
    g = play_game()
    
    m, tile, valid = g.send(None)#the first state
    if tile>3: tile=3 # when next>=3, we don't know it
    inp = get_input(m,tile) # the input for NN
    while valid:
        a_t = random.choice(valid)
        #if DEBUG: print(['UP','DOWN','LEFT','RIGHT'][a_t])
        # forward one step
        #if DEBUG: print('Moving...',end=' ')
        m1, tile1, valid1 = g.send(a_t)
        if tile1>3: tile1=3
        inp1 = get_input(m1,tile1)
        episode.append((m,tile,valid,inp,a_t,
                        m1,tile1,valid1,inp1,
                        to_score(m)))
        # update
        m,tile,valid,inp = m1,tile1,valid1,inp1
    g.close()
    scores.append(to_score(m)) # the final score when dead
    nums.append(m.max())

print(np.mean(scores))
print(np.max(nums))
