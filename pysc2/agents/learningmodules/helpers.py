
import logging
import random
from learningmodules.objects import *
from pysc2.lib import features

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

def getCountOfObjectOnScreen(obs, id, name):
  x, y = getCoordsOfObjectOnScreen(obs, id)
  if not name in SIZES:
    logging.info('No saved size for object ' + name + '. Total Pixels: ' + str(len(y)))
    return -1
  # logging.info(name +' '+ str(SIZES[name]))
  return int((len(y) + (SIZES[name] - 2)) / (SIZES[name]))

def getPointOnObjectOnScreen(obs, object):
	x, y = getCoordsOfObjectOnScreen(obs, object)
	if y.any():
		randomPoint = random.randint(0, len(y) - 1)
		return [x[randomPoint], y[randomPoint]]

def getCoordsOfObjectOnScreen(obs, object):
	unit_type = obs.observation['screen'][_UNIT_TYPE]
	unit_y, unit_x = (unit_type == object).nonzero()
	return (unit_x, unit_y)