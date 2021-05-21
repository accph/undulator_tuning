#!/usr/bin/env python2

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np
from functools import reduce

class Beamline():
    def __init__(self, geo, varElements, values=None):
        self.geo = geo
        self.varElements = varElements
        self.values = values

    def length(self):
        return sum(el.L for el in self.geo)

    def _set_values(self, values):
        if values is not None:
            for el, value in zip(self.varElements, values):
                el.set_var(value)

        if self.values is not None:
            for el, value in zip(self.varElements, self.values):
                el.set_var(value)

    def M(self, values=None):
        self._set_values(values)
        return reduce(np.dot, [el.M() for el in reversed(self.geo)])

    def twM(self, values=None):
        self._set_values(values)
        return reduce(np.dot, [el.twM() for el in reversed(self.geo)])

    def tw_transform(self, m1, m2):
        i1 = self.geo.index(m1)
        i2 = self.geo.index(m2)

        if i2 > i1:
            return np.dot( reduce( np.dot, [el.twM() for el in \
                reversed(self.geo[i1:i2])] ), m1.tw)
        else:
            return np.dot( reduce( np.dot, [el.twM() for el in \
                self.geo[i2:i1]]), m1.tw * np.array([1,-1,1,1,-1,1]))

    def beta_full(self, m):
        index = self.geo.index(m)
        if index == 0:
            tw = m.tw
        else:
            rev = np.array([1.,-1.,1.,1.,-1.,1.])
            tw = np.dot(reduce(np.dot, [el.twM() for el in self.geo[:index]]), \
                m.tw * rev) * rev

        bx = np.zeros(1+len(self.geo))
        by = np.zeros(1+len(self.geo))
        s = np.zeros(1+len(self.geo))
        bx[0] = tw[0]
        by[0] = tw[3]
        s[0] = 0.

        for i, el in enumerate(self.geo):
            s[i+1] = s[i]+el.L
            tw = np.dot(el.twM(), tw) 
            bx[i+1] = tw[0]
            by[i+1] = tw[3]
            

        return s, bx, by
