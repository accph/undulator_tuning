#!/usr/bin/env python2

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np

class Undulator():
    def __init__(self, L, lu, K, gamma, alpha=0., var_name='K'):
        self.L = L
        self.lu = lu
        self.K = K
        self.gamma = gamma
        self.alpha = alpha
        self.var_name = var_name


    def set_var(self, var):
        if self.var_name == 'K':
            self.K = var

    def csFunc(self, axis='x'):
        if axis == 'x':
            k = np.pi*np.sqrt(2*self.alpha)*self.K/(self.gamma*self.lu)
        elif axis == 'y':
            k = np.pi*np.sqrt(2*(1-self.alpha))*self.K/(self.gamma*self.lu)

        if k == 0:
            return np.array([1., self.L, 0., 1.])
        else:
            c = np.cos(self.L*k)
            s = 1./k*np.sin(self.L*k)
            return np.array([c, s, -(k**2)*s, c])

    def M(self):
        cx, sx, cpx, spx = self.csFunc(axis='x')
        cy, sy, cpy, spy = self.csFunc(axis='y')

        return np.array([[ cx,  sx,   0,   0], \
                         [cpx, spx,   0,   0], \
                         [  0,   0,  cy,  sy], \
                         [  0,   0, cpy, spy]])

    def twM(self):
        cx, sx, cpx, spx = self.csFunc(axis='x') \
                * np.array([1., 0.01, 100., 1.])
        cy, sy, cpy, spy = self.csFunc(axis='y') \
                * np.array([1., 0.01, 100., 1.])

        twMx = np.array([[   cx**2,      -2*cx*sx,   sx**2], \
                         [ -cx*cpx, cx*spx+cpx*sx, -sx*spx], \
                         [  cpx**2,    -2*cpx*spx,  spx**2]])

        twMy = np.array([[   cy**2,      -2*cy*sy,   sy**2], \
                         [ -cy*cpy, cy*spy+cpy*sy, -sy*spy], \
                         [  cpy**2,    -2*cpy*spy,  spy**2]])

        return np.block([[           twMx, np.zeros((3,3))], 
                         [np.zeros((3,3)),            twMy]])

if __name__ == '__main__':
    und = Undulator(33*12, 12, 0.69, 24.5, alpha=0.5)
    print('undulator: L=33*12, lu=12, K=0.69, gamma = 24.5, alpha=0.5')
    print(und.M())
    print(und.twM())

    import pdb; pdb.set_trace()
