from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import os.path
import random
import math
import logging

import numpy as np
import pandas as pd

from learningmodules.qlearning import QLearningTable
from learningmodules.possibleactions_protoss import *
from learningmodules.objects import *
from learningmodules.helpers import *

DATA_FILE = 'protest_agent_data'

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_HARVEST = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_CYBERNETICSCORE = actions.FUNCTIONS.Build_CyberneticsCore_screen.id
_BUILD_ASSIMILATOR = actions.FUNCTIONS.Build_Assimilator_screen.id
_BUILD_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_BUILD_SENTRY = actions.FUNCTIONS.Train_Sentry_quick.id
_BUILD_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_BUILD_ADEPT = actions.FUNCTIONS.Train_Adept_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


class Agent(base_agent.BaseAgent):

  def transformDistance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return [x - x_distance, y - y_distance]
    return [x + x_distance, y + y_distance]

  def transformLocation(self, x, y):
    if not self.base_top_left:
      return [64 - x, 64 - y]
    return [x, y]
  
  def splitAction(self, action_id):
    ai_action = ai_actions[action_id]
    x = 0
    y = 0
    if '_' in ai_action:
        ai_action, x, y = ai_action.split('_')
    return (ai_action, x, y)

  def __init__(self):
    super(Agent, self).__init__()
    self.qlearn = QLearningTable( actions = list(range(len(ai_actions))) )
    self.previous_action = None
    self.previous_state = None
    
    self.nex_y = None
    self.nex_x = None
    
    self.move_number = 0

    if os.path.isfile(DATA_FILE + '.gz'):
      logging.info('Loading learning data from file.')
      self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

  def step(self, obs):
    super(Agent, self).step(obs)

    # logging.info(obs.observation)

    if obs.last():
      reward = obs.reward
      self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
      self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
      self.previous_action = None
      self.previous_state = None
      self.move_number = 0
      return actions.FunctionCall(_NO_OP, [])

    if obs.first():
      player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
      self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
      self.nex_y, self.nex_x = getCoordsOfObjectOnScreen(obs, PROTOSS_NEXUS)

    pylon_count = getCountOfObjectOnScreen(obs, PROTOSS_PYLON, 'PROTOSS_PYLON')
    gateway_count = getCountOfObjectOnScreen(obs, PROTOSS_GATEWAY, 'PROTOSS_GATEWAY')
    assimilator_count = getCountOfObjectOnScreen(obs, PROTOSS_ASSIMILATOR, 'PROTOSS_ASSIMILATOR')
    cyberneticscore_count = getCountOfObjectOnScreen(obs, PROTOSS_CYBERNETICSCORE, 'PROTOSS_CYBERNETICSCORE')

    minerals = obs.observation['player'][1]
    gas = obs.observation['player'][2]
    army_supply = obs.observation['player'][_ARMY_SUPPLY]

    # STEP 1 --------------------------

    if self.move_number == 0:
      self.move_number += 1

      temp_state = [
        minerals,
        gas,
        pylon_count,
        gateway_count,
        assimilator_count,
        cyberneticscore_count,
        army_supply,
      ]
      state_length = len(temp_state)

      current_state = np.zeros(16 + state_length)
      for i in range(0, state_length):
        current_state[i] = temp_state[i]

      hot_squares = np.zeros(16)        
      enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
      for i in range(0, len(enemy_y)):
        y = int(math.ceil((enemy_y[i] + 1) / 16))
        x = int(math.ceil((enemy_x[i] + 1) / 16))
        hot_squares[((y - 1) * 4) + (x - 1)] = 1
      
      if not self.base_top_left:
        hot_squares = hot_squares[::-1]
      
      for i in range(0, 16):
        current_state[i + 4] = hot_squares[i]

      # logging.info(current_state)

      if self.previous_action is not None:
        self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

      rl_action = self.qlearn.choose_action(str(current_state))

      self.previous_state = current_state
      self.previous_action = rl_action

      ai_action, x, y = self.splitAction(self.previous_action)

      if ai_action == ACTION_BUILD_PYLON or ai_action == ACTION_BUILD_GATEWAY or ai_action == ACTION_BUILD_ASSIMILATOR or ai_action == ACTION_BUILD_CYBERNETICSCORE:
        probe = getPointOnObjectOnScreen(obs, PROTOSS_PROBE)
        if probe is not None:
          return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, probe])

      elif ai_action == ACTION_BUILD_ZEALOT or ai_action == ACTION_BUILD_SENTRY or ai_action == ACTION_BUILD_STALKER or ai_action == ACTION_BUILD_ADEPT:
        gateway = getPointOnObjectOnScreen(obs, PROTOSS_GATEWAY)
        if gateway is not None:
          return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, gateway])

      elif ai_action == ACTION_ATTACK:
        if _SELECT_ARMY in obs.observation['available_actions']:
          return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

      return actions.FunctionCall(_NO_OP, [])


    # STEP 2 --------------------------
    
    elif self.move_number == 1:
      self.move_number += 1
      
      ai_action, x, y = self.splitAction(self.previous_action)

      if ai_action == ACTION_HARVEST_GAS:
        if _HARVEST in obs.observation['available_actions']:
          assimilator = getPointOnObjectOnScreen(obs, PROTOSS_ASSIMILATOR)
          if assimilator is not None:
            return actions.FunctionCall(_HARVEST, [_NOT_QUEUED, assimilator])

      if ai_action == ACTION_BUILD_PYLON:
        if pylon_count < 5 and _BUILD_PYLON in obs.observation['available_actions']:
          if pylon_count == 0:
            target = self.transformDistance(round(self.nex_x.mean()), 8, round(self.nex_y.mean()), 6)
          elif pylon_count == 1:
            target = self.transformDistance(round(self.nex_x.mean()), 30, round(self.nex_y.mean()), 10)
          elif pylon_count == 2:
            target = self.transformDistance(round(self.nex_x.mean()), 0, round(self.nex_y.mean()), 30)
          elif pylon_count == 3:
            target = self.transformDistance(round(self.nex_x.mean()), 0, round(self.nex_y.mean()), 15)
          elif pylon_count == 4:
            target = self.transformDistance(round(self.nex_x.mean()), 20, round(self.nex_y.mean()), -6)
          return actions.FunctionCall(_BUILD_PYLON, [_NOT_QUEUED, target])

      elif ai_action == ACTION_BUILD_GATEWAY:
        if pylon_count > 0 and gateway_count < 3 and _BUILD_GATEWAY in obs.observation['available_actions']:
          if gateway_count == 0:
            target = self.transformDistance(round(self.nex_x.mean()), 20, round(self.nex_y.mean()), 11)
          elif gateway_count == 1:
            target = self.transformDistance(round(self.nex_x.mean()), 11, round(self.nex_y.mean()), 20)
          elif gateway_count == 2:
            target = self.transformDistance(round(self.nex_x.mean()), 31, round(self.nex_y.mean()), 0)
          return actions.FunctionCall(_BUILD_GATEWAY, [_NOT_QUEUED, target])

      elif ai_action == ACTION_BUILD_CYBERNETICSCORE:
        if cyberneticscore_count == 0 and _BUILD_CYBERNETICSCORE in obs.observation['available_actions']:
          target = self.transformDistance(round(self.nex_x.mean()), 20, round(self.nex_y.mean()), 0)
          return actions.FunctionCall(_BUILD_CYBERNETICSCORE, [_NOT_QUEUED, target])
      
      elif ai_action == ACTION_BUILD_ASSIMILATOR:
        if assimilator_count < 2 and _BUILD_ASSIMILATOR in obs.observation['available_actions']:
          target = getPointOnObjectOnScreen(obs, NEUTRAL_VESPENEGEYSER)
          if target is not None:
            return actions.FunctionCall(_BUILD_ASSIMILATOR, [_NOT_QUEUED, target])

      elif ai_action == ACTION_BUILD_ZEALOT:
        if _BUILD_ZEALOT in obs.observation['available_actions']:
          return actions.FunctionCall(_BUILD_ZEALOT, [_QUEUED])

      elif ai_action == ACTION_BUILD_STALKER:
        if _BUILD_STALKER in obs.observation['available_actions']:
          return actions.FunctionCall(_BUILD_STALKER, [_QUEUED])

      elif ai_action == ACTION_BUILD_SENTRY:
        if _BUILD_SENTRY in obs.observation['available_actions']:
          return actions.FunctionCall(_BUILD_SENTRY, [_QUEUED])

      elif ai_action == ACTION_BUILD_ADEPT:
        if _BUILD_ADEPT in obs.observation['available_actions']:
          return actions.FunctionCall(_BUILD_ADEPT, [_QUEUED])

      elif ai_action == ACTION_ATTACK:
        do_it = True
        
        if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == PROTOSS_PROBE:
            do_it = False
        
        if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == PROTOSS_PROBE:
            do_it = False
        
        if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
            x_offset = random.randint(-1, 1)
            y_offset = random.randint(-1, 1)
            
            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 7), int(y) + (y_offset * 7))])

    
      # STEP 3 --------------------------
      
    elif self.move_number == 2:
      self.move_number = 0

      ai_action, x, y = self.splitAction(self.previous_action)
              
      if ai_action == ACTION_BUILD_GATEWAY or ai_action == ACTION_BUILD_ASSIMILATOR or ai_action == ACTION_BUILD_PYLON or ai_action == ACTION_BUILD_CYBERNETICSCORE:
        if _HARVEST in obs.observation['available_actions']:
          minerals = getPointOnObjectOnScreen(obs, NEUTRAL_MINERALFIELD)
          if minerals is not None:
            return actions.FunctionCall(_HARVEST, [_QUEUED, minerals])

    return actions.FunctionCall(_NO_OP, [])
