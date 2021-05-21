#!/usr/bin/env python2

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np

class Mark():
    def __init__(self, tw=None):
        self.L = 0.
        self.tw = tw

    def M(self):
        return np.eye(4)

    def twM(self):
        return np.eye(6)

    def set_tw(ax, bx, ay, by):
        self.tw = np.array([bx, ax, (1.+ax**2)/bx, \
                            by, ay, (1.+ay**2)/by ])
