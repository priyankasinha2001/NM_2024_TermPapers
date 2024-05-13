#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sympy
import scipy.integrate as integrate
import scipy.constants as constants
from scipy.misc import derivative


# In[ ]:





# ## TaylorT4 Orbital Evolution(3 PN order)

# In[381]:


m1=5
m2=5
M =m1+m2
mu= m1*m2/M
eta= mu/M
pi= np.pi
gamma= 0.577216


# In[382]:


def der3(t,x):
    return 64/5*eta/M *x**5 *(1- (743/336 + 11/4*eta)*x + 4*pi*x**(3/2) - (34103/18144 - 13661/2016 *eta - 59/18 *eta**2)*x**2 - 
                             (4159/672 + 189/8 *eta)*pi*x**(5/2)  + ((16447322263/139708800) + 16/3 *(pi**2) - 1712/105*gamma - 856/105*np.log(16*x)
                                + (41/48 *pi**2 - 134543/7776)*eta - 94403/3024 *(eta**2) - 775/324 *(eta**3))* x**3
                             -(4415/4032 - 358675/6048*eta - 91495/1512 *eta**2)*pi*x**(7/2))
    


# In[383]:


def der4(t,x):
    return (x**(3/2))/M


# In[384]:


def coupled2(t, y_vec):
    x,phi = y_vec
    return [der3(t,x), der4(t,x)]


# In[385]:


inits= [0.1, 0]  
tmax,Nt= 12000, 50000
t_arr = np.linspace(0, tmax, Nt)
t0=0


# In[386]:


solve1 = integrate.solve_ivp(coupled2, t_span = [t0,tmax], y0 = inits, t_eval=t_arr)


# In[387]:


x = solve1.y[0]
phi = solve1.y[1]


# In[388]:


plt.plot((solve1.t), x, lw=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('x(t) vs t')
plt.legend()
plt.grid()


# In[389]:


plt.plot(solve1.t, phi, lw=2)
plt.xlabel('t')
plt.ylabel(r'$\varphi(t)$')
plt.title('phi(t) vs t')
plt.grid()


# In[390]:


h_22 = -8*np.sqrt(5/pi)*mu*np.exp(-2*1j*phi)*x*(1-(107/42 -55/42 *eta)*x +(2*pi + 6*1j*np.log(x))*x**(3/2)-(2173/1512 + 1096/216 *eta - 2047/1512 *eta**2)*x**2 -
    ((107/21 -34/21 *eta)*pi + 24*1j*eta + 1j*(107/7 -34/7 *eta)*np.log(x))*x**(5/2)
        +(27027409/646800 - 856/105*gamma +2/3*pi**2 -1712/105 *np.log(2) - 428/105*np.log(x)
    -18*(np.log(x))**2 - (278185/33264 - 41/96 *pi**2)*eta - 20261/2772 *eta**2 + 114635/99792 *eta**3 + 1j* 428/105*pi +12*1j*pi*np.log(x))*x**3)


# In[391]:


#h_2_2 = -8*np.sqrt(5/pi)*mu*np.exp(2*1j*phi)*x*(1-(107/42 -55/42 *eta)*x +(2*pi + 6*(-1j)*np.log(x))*x**(3/2)-(2173/1512 + 1096/216 *eta - 2047/1512 *eta**2)*x**2 -
#    ((107/21 -34/21 *eta)*pi + 24*(-1j)*eta + (-1j)*(107/7 -34/7 *eta)*np.log(x))*x**(5/2)
#        +(27027409/646800 - 856/105*gamma +2/3*pi**2 -1712/105 *np.log(2) - 428/105*np.log(x)
#    -18*(np.log(x))**2 - (278185/33264 - 41/96 *pi**2)*eta - 20261/2772 *eta**2 + 114635/99792 *eta**3 + (-1j)* 428/105*pi +12*(-1j)*pi*np.log(x))*x**3)


# In[392]:


theta= np.pi/3
Phi= -np.pi/8

Y_22p = 1/8 * np.sqrt(5/pi)*(1+ np.cos(theta))*np.exp(2*1j*Phi)
Y_22m = 1/8 * np.sqrt(5/pi)*(1- np.cos(theta))*np.exp(-2*1j*Phi)


# In[393]:


h = Y_22p*h_22 + Y_22m*h_22

h_plus= h.real
h_cross= h.imag


# In[394]:


tmax,Nt= 12000, 23833
t_ar = np.linspace(0, tmax, Nt)


# In[395]:


plt.plot(solve1.t,h_plus, label='h_plus')
plt.plot(solve1.t, h_cross, label='h_cross')
plt.xlabel('t')
plt.ylabel('h(t)')
#plt.ylim(-2,2)
plt.grid()
plt.legend()


# In[ ]:





# ## 1 PN order

# In[400]:


tmax,Nt= 12000, 50000
t_arr = np.linspace(0, tmax, Nt)
t0=0
x0= [0.1]


# In[401]:


m1= 5
m2= 5
m= m1+ m2
mu= m1*m2/m
eta=mu/m

def F(x):
    return 32/5* (eta**2)* x**5

def E(x):
    return -1/2 * eta* x

def dE_dx(x):
    return -1/2 *eta


def der1(t,x):
    return -F(x)/(m*dE_dx(x))
    


# In[365]:


def der2(t,x):
   return (x**(3/2))/m

phi_0= [0]


# In[366]:


inits= [0.1, 0]  

def coupled1(t, y_vec):
    x,phi = y_vec
    return [der1(t,x), der2(t,x)]


# In[367]:


solve2 = integrate.solve_ivp(coupled1,t_span = [t0,tmax], y0 = inits, t_eval = t_arr)


# In[368]:


x_t = solve2.y[0]
phi_t = solve2.y[1]


# In[372]:


plt.plot(solve2.t, x_t, lw=2)
plt.xlabel('time')
plt.ylabel('x(t)')
plt.grid()


# In[373]:


plt.plot(solve2.t, phi_t, lw=2)
plt.xlabel('time')
plt.ylabel(r'$\varphi(t)$')
plt.grid()


# In[376]:


h_22_1= -8*np.sqrt(5/pi)*mu*np.exp(-2*1j*phi_t)*x_t


# In[378]:


h_1 = Y_22p*h_22_1 + Y_22m*h_22_1

h_p= h_1.real
h_c= h_1.imag


# In[379]:


plt.plot(solve2.t,h_p, label='h_p')
plt.plot(solve2.t, h_c, label='h_c')
plt.xlabel('t')
plt.ylabel('h(t)')
#plt.ylim(-2,2)
plt.grid()
plt.legend()


# In[ ]:





# In[ ]:





# ### compare 3PN and 1PN order

# In[419]:


plt.plot(solve2.t/7813,h_p,  label='hp(1PN)')
plt.plot(solve1.t/5719,h_plus, label='hp(3PN)')
plt.xlabel('time')
plt.legend()
plt.grid()


# In[420]:


plt.plot(solve2.t/7813,h_c,  label='hc(1PN)')
plt.plot(solve1.t/5719,h_cross, label='hc(3PN)')
plt.xlabel('time')
plt.legend()
plt.grid()


# In[418]:


solve2.t[-1]


# In[417]:


solve1.t[-1]


# In[ ]:




