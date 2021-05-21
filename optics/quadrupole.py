#!/usr/bin/env python2

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np
from scipy.interpolate import CubicSpline

class Quadrupole():
    def __init__(self, L, G, gamma, file=None, var_name='G'):
        self.L = L
        self.G = G
        self.gamma = gamma
        self.file = file
        self.var_name = var_name

        if file is not None:
            data = np.loadtxt(file, dtype='float64')
            self._Gspline = CubicSpline(data[:,0], data[:,1])

    def set_var(self, var):
        if self.var_name == 'G':
            self.G = var

    def csFunc(self, h, g, axis='x'):
        pc = 0.511e6*np.sqrt(self.gamma**2 - 1)
        k = np.sign(g)*np.sqrt(300.*abs(g)/pc)
        #import pdb; pdb.set_trace()
        if k == 0:
            return np.array([1., h, 0., 1.])
        elif (k>0 and axis=='x') or (k<0 and axis=='y'):
            c = np.cos(h*abs(k))
            s = 1./abs(k)*np.sin(h*abs(k))
            return np.array([c, s, -(k**2)*s, c])
        elif (k<0 and axis=='x') or (k>0 and axis=='y'):
            c = np.cosh(h*abs(k))
            s = 1./abs(k)*np.sinh(h*abs(k))
            return np.array([c, s, (k**2)*s, c])

    def M(self):
        if self.file is None:
            #import pdb; pdb.set_trace()
            cx, sx, cpx, spx = self.csFunc(self.L, self.G, axis='x')
            cy, sy, cpy, spy = self.csFunc(self.L, self.G, axis='y')

            return np.array([[ cx,  sx,   0,   0], \
                             [cpx, spx,   0,   0], \
                             [  0,   0,  cy,  sy], \
                             [  0,   0, cpy, spy]])
        else:
            pass

    def twM(self):
        if self.file is None:
            cx, sx, cpx, spx = self.csFunc(self.L, self.G, axis='x') \
                * np.array([1., 0.01, 100., 1.])
            cy, sy, cpy, spy = self.csFunc(self.L, self.G, axis='y') \
                * np.array([1., 0.01, 100., 1.])

            twMx = np.array([[   cx**2,      -2*cx*sx,   sx**2], \
                             [ -cx*cpx, cx*spx+cpx*sx, -sx*spx], \
                             [  cpx**2,    -2*cpx*spx,  spx**2]])

            twMy = np.array([[   cy**2,      -2*cy*sy,   sy**2], \
                             [ -cy*cpy, cy*spy+cpy*sy, -sy*spy], \
                             [  cpy**2,    -2*cpy*spy,  spy**2]])

            return np.block([[           twMx, np.zeros((3,3))], 
                             [np.zeros((3,3)),            twMy]])
        else:
            pass  

if __name__ == '__main__':
    pc = 0.511e6*np.sqrt(24.5**2-1)
    quad = Quadrupole(16, 4.90237618e-4/300*pc, 24.5)
    print(quad.M())
    print()
    print(quad.twM())
