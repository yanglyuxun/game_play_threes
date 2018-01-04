#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threes API for threesjs.com
"""
from bs4 import BeautifulSoup
import os, time
import numpy as np

from selenium import webdriver # for show the page
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from game.python_threes import play_game
# this is a python version of Threes by Robert Xiao
# I modified some places to make use of it

game_url = 'file://'+os.path.join(os.getcwd(),'game/threes.html')
# this is the game from http://threesjs.com
ph_path = './phantomjs-2.1.1-linux-x86_64/bin/phantomjs'
col_code = {'number':3,'blue':1, 'red':2} # the next color to the number

def get_state(html):
    '''input html text (for BeautifulSoup) and return
    the current state'''
    b = BeautifulSoup(html,"html5lib")
    board = np.zeros((4,4),dtype=np.int)
    for t in b.find_all(attrs={'class':'tile'}):
        col = t.get('class')[-1]
        xy = t.get('data-coords')
        if xy:
            x,y = [int(i) for i in xy]
            board[x,y] = int(t.get_text())
        else:
            next_n = col_code[col]
    return [board.reshape((1,4,4,1)).copy(), 
            np.array([next_n]).reshape((1,1)).copy()], calc_score(board)

def calc_score(board):
    score = 0
    for k in board.flatten():
        if k>2: # 1 & 2 do not count
            score += np.int(3 ** (np.log2(k/3)+1))
    return score

def get_max_number(state):
    '''input the state and output the max number by now'''
    board = state[0][:,:,0]
    return board.max()

class threes_API(object):
    '''the API to control the js game'''
    def __init__(self, see=True):
        '''see: weather to use chrome to see the process'''
        if see:
            self.d = webdriver.Chrome()
        else:
            self.d = webdriver.PhantomJS(executable_path=ph_path)
        self.d.get(game_url)
        self.max_score = 0
        self.max_num = 0
    def if_over(self):
        end = self.d.find_elements_by_class_name('endgame')
        return len(end)!=0
    def restart(self):
        print('Restarting...',end=' ')
        self.last_score = int(self.d.find_element_by_class_name('score').text)
        self.last_num = get_max_number(get_state(self.d.page_source)[0])
        if self.max_num <  self.last_num:
            self.max_num = self.last_num
        if self.max_score <self.last_score:
            self.max_score = self.last_score
        while True:
            self.d.refresh()
            time.sleep(0.1)
            if not self.if_over(): break
        print('Done.')
    def first_step(self):
        '''the first step to give out'''
        if self.if_over(): self.restart()
        self.state, self.score = get_state(self.d.page_source)
        return self.state
    def next_step(self,action,wait=0.3):
        '''perform an action and return the next state and rewards
        0: UP
        1: DOWN
        2: LEFT
        3: RIGHT'''
        act_ch = ActionChains(self.d)
        if action == 0:
            act_ch.key_down(Keys.ARROW_UP).perform()
        elif action == 1:
            act_ch.key_down(Keys.ARROW_DOWN).perform()
        elif action == 2:
            act_ch.key_down(Keys.ARROW_LEFT).perform()
        elif action == 3:
            act_ch.key_down(Keys.ARROW_RIGHT).perform()
        else:
            raise 'Wrong action!'
        time.sleep(wait) # wait for the action to be completed
        state1, score1 = get_state(self.d.page_source)
        die = self.if_over()
        if die or (self.state[0]==state1[0]).all():
            reward = - self.score # if over or invalid, punish
        else:
            reward = score1 - self.score # valid move reward 100
        if die: 
            self.restart()
            state1, score1 = get_state(self.d.page_source)
        self.state, self.score = state1, score1 # update the score
        return state1, reward, die





def to_val(x):
    ''' from m to value x'''
    return np.where(x <= 3, x, 3*2.0**(x-3))

def to_score(x):
    '''from m to score x'''
    return np.where(x < 3, 0, 3.0**(x-2))

class Python_threes_API(object):
    ''' to control the python version of Threes'''
    def __init__(self):
        self.game = play_game()
        self.max_score = 0
        self.max_num = 0 
    def first_step(self):
        m, tileset, valid = self.game.send(None)
        self.state = [m.reshape((1,4,4,1))/m.max(),
                      np.array(tileset).reshape((1,1))]
        self.score = to_score(m).sum()
        self.last_num = m.max()
        self.valid = valid
        return self.state.copy()
    def restart(self):
        self.last_score = self.score
        #self.last_num = self.state[0].max() ## it is coded num!
        self.max_score = np.max((self.max_score, self.last_score))
        self.max_num = np.max((self.max_num, self.last_num))
        self.game = play_game()
        self.first_step()
    def next_step(self,move):
        if move not in self.valid: #punish invalid move
            return self.state.copy(), - self.score, False
        m, tileset, valid = self.game.send(move)
        score1 = to_score(m).sum()
        self.last_num = m.max()
        if valid:
            reward = score1 - self.score # valid
            die = False
            self.score = score1
            self.state = [m.reshape((1,4,4,1))/m.max(),
                  np.array(tileset).reshape((1,1))]
            self.valid = valid.copy()
        else:
            reward = - self.score # die, punish
            self.restart() # it updates all
            die = True
        return self.state.copy(), reward, die
