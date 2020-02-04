# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:10:26 2019
@author: benji
"""
import numpy as np
import matplotlib.pyplot as plt
print ("Benjamin Moore")
"""assingment 4
Aim: to study the numerical solution to ODE
(1 + t)x + 1 − 3t + t^2
using the 3 numerical schemes, and show the varying accuracy of them"""
"""solution of such an ODE can be understood by plotting the so-called direction
field. This consists of small arrows, of slope dx/dt in the t-x plane"""

#PART 1
"""Produce such a direction field for t ranging from 0 to 5 and x from -3 to +3 by
evaluating dx/dt at 25 × 25 grid points. Python commands np.meshgrid(,) and
ax.quiver(,,,) are useful for this."""
def dxdt(t,x):
    return ((1+t)*x)+1-(3*t)+t**2
x=np.linspace(-3, 3, 25)
t=np.linspace(0, 5, 25)
t, x = np.meshgrid(t, x)

#25x25 grid points
#arrows are tangent to the solutions
plt.figure()
plt.quiver(t,x,1,dxdt(t,x), scale=50)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('Direction field\slope field\vector field of ODE')
plt.ylim(-3.0,3.0)
plt.xlim(0,5.0)
plt.show()


#PART 2
"""Pick as a starting value x(t = 0) = 0.0655 and solve eqn.(1) using the simple
Euler method with step size 0.04. (Feel free to use some of the python code
presented in the lecture (and available on Blackboard) for solving ODEs.) Plot
your solution together with the direction field. """

stepsize = 0.04
initial = 0.0
end = 5.0
X_zero = 0.0655
t_zero = 0.0
def seuler(t,x,nsize):
     return x + nsize*dxdt(t,x)

t1 = np.arange(initial,end,stepsize)
#from range from first step to last-125 steps
simple_euler = np.zeros(125)
simple_euler[0] = X_zero

for i in range(1,125):
   simple_euler[i] = seuler(t1[i-1], simple_euler[i-1], stepsize)

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t1,simple_euler,color='GREEN')
plt.ylim(-3,3)
plt.title('Simple Euler method, step size=0.04')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc="upper right")
plt.show()
 


#PART 3
""" Repeat this calculation using both improved Euler method and Runge-Kutta
method. (Again feel free to use part of the code provided on Blackboard.)
How do your new numerical solutions behave? Reduce the step size to 0.02,
what do you observe now? Can you see the benefit of using accurate integration
schemes?"""


def ieuler(t,x,nsize):
                return x + 0.5*nsize*(dxdt(t,x) + dxdt(t + nsize,x+nsize*dxdt(t,x)))

def rk(t,x,nsize):
                k1 = dxdt(t,x)
                k2 = dxdt(t+ 0.5*nsize,x + 0.5*nsize*k1)
                k3 = dxdt(t + 0.5*nsize,x + 0.5*nsize*k2)
                k4 = dxdt(t + nsize,x + nsize*k3,)
                return x + nsize/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)


ieul = np.zeros(125)
ruku = np.zeros(125)

ieul[0] = X_zero
ruku[0] = X_zero               
for i in range(1,125):
   ieul[i] = ieuler(t1[i-1], ieul[i-1], stepsize)
   ruku[i] = rk(t1[i-1], ruku[i-1], stepsize)

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t1,ieul,color='BLUE',label='Improved Euler method')
plt.plot(t1,ruku,color='RED',label='Runge-Kutta method')
plt.ylim(-3,3)
plt.title('Improved numerical methods, step size=0.04')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t1,simple_euler,color='GREEN',label='Euler method')
plt.plot(t1,ieul,color='BLUE',label='Improved Euler method')
plt.plot(t1,ruku,color='RED',label='Runge-Kutta method')
plt.ylim(-3,3)
plt.title('All three numerical methods, step size=0.04')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc="upper right")
plt.show()

stepsize2 = 0.02
t2 = np.arange(initial,end,stepsize2)
#having step size, doubles steps
seul1 = np.zeros(250)
ieul1 = np.zeros(250)
ruku1 = np.zeros(250)

seul1[0] = X_zero
ieul1[0] = X_zero
ruku1[0] = X_zero

for i in range(1,250):
    seul1[i] = seuler(t2[i-1], seul1[i-1], stepsize2)
    ieul1[i] = ieuler(t2[i-1], ieul1[i-1], stepsize2)
    ruku1[i] = rk(t2[i-1], ruku1[i-1], stepsize2)

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t2,seul1,color='GREEN',label='Euler method')
plt.plot(t2,ieul1,color='BLUE',label='Improved Euler method')
plt.plot(t2,ruku1,color='RED',label='Runge-Kutta method')
plt.ylim(-3,3)
plt.title('All three numerical methods, step size=0.02')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t2,seul1,color='black',label='step size=0.02')
plt.ylim(-3,3)
plt.title('Comparing Simple Euler method solutions for step size=0.02 and 0.04')
plt.plot(t1,simple_euler,color='Green',label='step size=0.04')
plt.legend(loc='upper right')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.show()
 
plt.quiver(t, x, 1, dxdt(t,x), scale=50)
plt.plot(t2,ieul1,color='black', label='step size 0.02')
plt.plot(t1,ieul,color='green', label='step size 0.04')
plt.ylim(-3,3)
plt.title('Comparing Improved Euler method for step size=0.02 and 0.04')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc='upper right')
plt.show()
 
plt.figure()
plt.quiver(t, x, 1, dxdt(t,x), scale=50)
plt.plot(t2,ruku1,color='black', label='step size 0.02')
plt.plot(t1,ruku,color='green', label='step size 0.04')
plt.ylim(-3,3)
plt.title('Comparing Runge-Kutta method for step size=0.02 and 0.04')
plt.ylabel("x")
plt.xlabel("t")
plt.legend(loc='upper right')
plt.show()


stepsize3=0.01

t1 = np.arange(initial,end,stepsize3)
#from range from first step to last-125 steps
simple_euler = np.zeros(500)
simple_euler[0] = X_zero

for i in range(1,500):
   simple_euler[i] = seuler(t1[i-1], simple_euler[i-1], stepsize)

plt.figure()
plt.quiver(t, x, 1, dxdt(t,x),scale=50)
plt.plot(t1,simple_euler,color='GREEN', label='step siz=0.01')
plt.plot(t2,seul1,color='black',label='step size=0.02')
plt.ylim(-3,3)
plt.title('Simple Euler method, step size=0 0.01 and 0.02')
plt.ylabel("x(t)")
plt.xlabel("t")
plt.legend(loc="upper right")
plt.show()
 






