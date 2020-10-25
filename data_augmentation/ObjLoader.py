#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:25:25 2020

@author: franciscapessanha
"""

import numpy as np

class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.norm_coords = []

        self.vertex_index = []
        self.normal_index = []

        self.faces = []

    def load_model(self, file):
        for line in open(file, 'r'):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                self.vert_coords.append(values[1:4])
            if values[0] == 'vn':
                self.norm_coords.append(values[1:4])


            if values[0] == 'f':
                #f = np.hstack([i.split('//')[0] for i in values[1:]])
                self.faces.append(line)
