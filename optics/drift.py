#!/usr/bin/env python2

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np

class Drift():
    def __init__(self, L, var_name='L'):
        self.L = L
        self.var_name = var_name

    def set_var(self, var):
        if self.var_name == 'L':
            self.L = var

    def M(self):
        L = self.L
        return np.array([[1, L, 0, 0], \
                         [0, 1, 0, 0], \
                         [0, 0, 1, L], \
                         [0, 0, 0, 1]])

    def twM(self):
        L = 0.01*self.L
        return np.array([[1, -2*L, L**2,    0,    0,    0], \
                         [0,    1,   -L,    0,    0,    0], \
                         [0,    0,    1,    0,    0,    0], \
                         [0,    0,    0,    1, -2*L, L**2], \
                         [0,    0,    0,    0,    1,   -L], \
                         [0,    0,    0,    0,    0,    1]])
