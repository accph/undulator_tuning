#!/usr/bin/env python3

# matveev.a.s@yandex.ru
# 2018 Anton Matveev

import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import solve, inv
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
import os

from optics.drift import Drift
from optics.quadrupole import Quadrupole
from optics.undulator import Undulator
from optics.beamline import Beamline
from optics.mark import Mark

data = np.loadtxt(os.path.abspath('source/I_K.dat'), dtype='float64')
K_spline = CubicSpline(data[:,0],data[:,1])
i_spline = CubicSpline(data[:,1],data[:,0])

data = np.loadtxt(os.path.abspath('source/K_b0.dat'), dtype='float64')
b0 = CubicSpline(data[:,0],data[:,1])


gamma = 11.911/0.511
pc = 0.511e6*np.sqrt(gamma**2 - 1)

i0 = [1.732, -3.478, 2.142, 1.944, -3.69, 3.6, -2.588, 1.659]
Iund0 = 1526

ds = [Drift(L) for L in [34.19, 32.86, 29.94, 26.85, \
                             31.75, 32.0, 40.84, 39.25]]
du = Drift(65.45)
und = Undulator(366., 12., K_spline(Iund0), gamma, alpha=0.5)
qs = [Quadrupole(16., 189./16.*i, gamma) for i in i0]
mark = Mark(np.array([b0(K_spline(Iund0)), 0., 1./b0(K_spline(Iund0))]*2))
geo = [qs[0],ds[0],qs[1],ds[1],qs[2],ds[2],qs[3],ds[3],und,du, mark, \
       du,und,ds[4],qs[4],ds[5],qs[5],ds[6],qs[6],ds[7],qs[7]]

line = Beamline(geo, varElements=qs[1:-1]+[und])

m0 = line.M()
m0_inv = inv(m0)
s0, bx0, by0 = line.beta_full(mark)

def fun(Is, Iu):
    m = line.M(np.append(Is*189./16., K_spline(Iu)))
    d = np.dot(m, m0_inv) - np.eye(4)
    return np.array([d[1,0],d[1,1],d[0,1],d[2,2],d[3,2],d[2,3]])

def fun_d(Is, Iu, dx=1e-4):
    res = np.empty([6,6])
    for i in range(6):
        Is2 = Is.copy()
        Is2[i] += dx
        res[:,i] = (fun(Is2,Iu)-fun(Is,Iu))/dx
    return res

def fun_dK(Is, Iu, dx=1e-4):
    return (fun(Is,Iu+dx)-fun(Is,Iu))/dx

def x_d(Is, Iu):
    return solve(fun_d(Is, Iu), -fun_dK(Is, Iu))



if __name__ == '__main__':
    t = np.linspace(Iund0, 1200, 10)
    res = odeint(x_d, i0[1:-1], t)

    f, axarr = plt.subplots(2, sharex=True)
    for i in range(6):
        axarr[0].plot(t, list(zip(*res))[i],'.-')

    axarr[1].plot(t, [np.linalg.norm(fun(Is, t1)) for Is, t1 in zip(res,t)], '.-')
    axarr[1].set(ylabel='obj fun')

    axarr[1].set(xlabel='I und')
    f.subplots_adjust(hspace=0)
    plt.show()


    line._set_values(np.append(res[-1]*189./16., K_spline(t[-1])))
    mark.tw = np.array([b0(K_spline(t[-1])), 0., 1./b0(K_spline(t[-1]))]*2)
    s, bx, by = line.beta_full(mark)

    plt.plot(0.01*s0, bx0, 'b^-', label=r'$\beta0_x$')
    plt.plot(0.01*s0, by0, 'r^-', label=r'$\beta0_y$')
    plt.plot(0.01*s, bx, 'b.--', label=r'$\beta_x$')
    plt.plot(0.01*s, by, 'r.--', label=r'$\beta_y$')
    plt.xlabel('s, (m)')
    plt.ylabel(r'$\beta$, (m)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1),  ncol=2)
    plt.show()

