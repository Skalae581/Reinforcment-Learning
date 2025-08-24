# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:07:08 2025

@author: TAKO
"""

import gymnasium as gym
 
import pickomino_env
ENV_NAME = "Pickomino-v0"  

env=gym.make("Pickomino-v0", num_players=2)