import numpy as np
import matplotlib.pyplot as plt
from math import *
from mpl_toolkits.mplot3d import Axes3D

global H_type
global G_type

def H(x):
    if H_type == 1:
        return exp(-sin(2*pi*x))/20
    elif H_type == 2:
        return x**2/10
    elif H_type == 3:
        return 0
    elif H_type == 4:
        return 2
    
def Hx(x): # dH/dx
    if H_type == 1:
        return -2*pi*cos(2*pi*x)*exp(-sin(2*pi*x))/20
    elif H_type == 2:
        return x/5
    elif H_type == 3:
        return 0
    elif H_type == 4:
        return 0

def G(x):
    if G_type == 1:
        return cos(2*pi*x)
    elif G_type == 2:
        return x**2
    elif G_type == 3:
        # return exp(sin(10*x))
        return -1
    elif G_type == 4:
        return 0.5

def lx(t): # left Dirichlet boundary condition; x = 0
    # return sin(2*pi*t)
    return 0

def rx(t): # right Dirichlet boundary condition; x = 1
    # return t*(1-t)
    return 0

def ivc(x): # initial value condition; want this to equal lx(0) and rx(0) at boundaries
    # return sin(2*pi*x)+(x)*(1-x)/2
    return x*(1-x)
    # return x*(np.pi-x)



def data_1d_u():
    '''
    Generates data for u using FDM
    Generates this for a domain of [0,T] = [0,2] and Ω = [0,1]
    with time split into 20000 evenly spaced intervals and space 100
    '''

    T = 2
    X = 1

    dt = T / 20000 # 0.0001
    dx = X / 100 # 0.01

    u = np.zeros((20001, 101)) # shape (20001, 101); indexed from u[0][0] to u[20000][100]

    for k in range(101):
        u[0][k] = ivc(dx * k)

    for j in range(20000):
        for k in range(1, 100):
            x = dx*k
            # see notes for FDM implementation; this is well-defined
            u[j+1][k] = u[j][k] + dt/dx/dx*H(x)*(u[j][k+1]-2*u[j][k]+u[j][k-1]) +\
                        Hx(x)*(u[j][k+1]-u[j][k-1])/2/dx*dt + G(x)*u[j][k]*dt

        u[j+1][0]=lx((j+1)*dt) # t = dt*(j+1)
        u[j+1][100]=rx((j+1)*dt)

    return u

def data_1d_u_t(u):
    u_t = np.zeros((20001, 101))

    dt = 2 / 20000
    dx = 1 / 100

    # central finite difference for interior points
    # backward/forward differences for boundary points
    for k in range(101):
        for j in range(1, 20000):
            u_t[j][k] = (u[j+1][k] - u[j-1][k]) / 2 / dt

        u_t[0][k] = (u[1][k] - u[0][k]) / dt
        u_t[20000][k] = (u[20000][k] - u[19999][k]) / dt

    return u_t
  
def data_1d_u_x(u):
    u_x = np.zeros((20001, 101))

    dt = 2 / 20000
    dx = 1 / 100

    # central finite difference for interior points
    # backward/forward differences for boundary points
    for j in range(20001):
        for k in range(1, 100):
            u_x[j][k] = (u[j][k+1] - u[j][k-1]) / 2 / dx

        u_x[j][0] = (u[j][1] - u[j][0]) / dx
        u_x[j][100] = (u[j][100] - u[j][99]) / dx

    return u_x
  
def data_1d_u_xx(u):
    u_xx = np.zeros((20001, 101))

    dt = 2 / 20000
    dx = 1 / 100

    # central second finite difference for interior points
    # backward/forward second differences for boundary points
    for j in range(20001):
        for k in range(1, 100):
            u_xx[j][k] = (u[j][k+1] - 2*u[j][k] + u[j][k-1]) / dx / dx

        u_xx[j][0] = (u[j][2] - 2*u[j][1] + u[j][0]) / dx / dx
        u_xx[j][100] = (u[j][100] - 2*u[j][99] + u[j][98]) / dx / dx
        # u_xx[j][0] = (2*u[j][0] - 5*u[j][1] + 4*u[j][2] - u[j][3]) / dx / dx / dx
        # u_xx[j][100] = (2*u[j][100] - 5*u[j][99] + 4*u[j][98] - u[j][97]) / dx / dx / dx

    return u_xx




### FOR TYPE 4:

def exact_u(x,t): # calculates exact value of u given an x and t
    total = 0
    n_sums = 30
    
    for n in range(1, n_sums+1):
        total += -4 * np.sin(n*x) * np.exp(t*(-2*n**2+0.5)) * ((-1)**n-1) / np.pi / n**3
        
    return total

def exact_u_t(x,t): # calculates exact value of u_t given an x and t
    total = 0
    n_sums = 30
    
    for n in range(1, n_sums+1):
        total += -4 * np.sin(n*x) * np.exp(t*(-2*n**2+0.5)) * ((-1)**n-1) / np.pi / n**3\
                 * (-2 * n**2 + 0.5)
        
    return total

def exact_u_x(x,t): # calculates exact value of u_x given an x and t
    total = 0
    n_sums = 30
    
    for n in range(1, n_sums+1):
        total += -4 * np.cos(n*x) * np.exp(t*(-2*n**2+0.5)) * ((-1)**n-1) / np.pi / n**3\
                 * n
        
    return total

def exact_u_xx(x,t): # calculates exact value of u_xx given an x and t
    total = 0
    n_sums = 30
    
    for n in range(1, n_sums+1):
        total += -4 * np.sin(n*x) * np.exp(t*(-2*n**2+0.5)) * ((-1)**n-1) / np.pi / n**3\
                 * (-n**2)
        
    return total
























# same as data_1d() but add noise
def data_1d_sol_noise(sigma=0.1,T=1,X=1,H=H,Hx=Hx,G=G):
    T = 2
    X = 1
    dx = 0.01
    dt = 0.0001

    u = np.zeros((int(T/dt)+1, int(X/dx)+1))

    for k in range(int(X/dx)+1):
        u[0][k] = sin(2*pi*dx*k)+(dx*k)*(1-dx*k)/2


    for j in range(int(T/dt)):
        for k in range(1, int(X/dx)):
            x = dx*k
            u[j+1][k] = u[j][k]+dt/dx/dx*H(x)*(u[j][k+1]-2*u[j][k]+u[j][k-1])+\
                        Hx(x)*(u[j][k+1]-u[j][k-1])/2/dx*dt+G(x)*u[j][k]*dt

        # u[j+1][0]= 2*u[j+1][1]-u[j+1][2]
        # u[j+1][int(X/dx)-1]= 2*u[j+1][int(X/dx)-2]-u[j+1][int(X/dx)-3]

        # u[j+1][0]=0
        # u[j+1][int(X/dx)]=0

        u[j+1][0] = lx((j+1)*dt)
        u[j+1][int(X/dx)] = rx((j+1)*dt)

    u = u + np.random.rand(int(T / dt)+1, int(X / dx)+1) * sigma
    return u


if __name__ == '__main__':
    X = 1
    dx = 0.01
    T = 2
    XX = np.linspace(0, 1, int(X/dx)+1)
    Hxx= [H(x) for x in XX]
    Gxx = [G(x) for x in XX]

    ### visualize H(x)
    # plt.plot(XX,Hxx,label='H(x)')
    # plt.legend()
    # plt.show()

    ### visualize G(x)
    # plt.plot(XX,Gxx,label='G(x)')
    # plt.legend(loc=1)
    # plt.show()

    # u1 = data_1d()
    # print(np.sqrt(np.mean(u1*u1)))

    ### visualize u across various time points on a graph of u against x
    u1 = data_1d_noise(sigma=0.5)
    for i in range(11): # i from 0 to 10
        j = i*2000 # plot x points when j = [0, 2000, ..., 20000]
        t = i/10*T # or, the time points t = [0, 0.2, ..., 2.0]
        # plt.scatter(XX, u1[j], label='t={}'.format(t),s=5)
        plt.plot(XX, u1[j], label='t={}'.format(t))
    plt.legend()
    plt.show()

    ### visualize u in a 3d graph of u against x and t
    figure = plt.figure()
    axes = Axes3D(figure)
    XX = np.linspace(0, 1, 101)
    TT = np.linspace(0, 2, 20001)
    XX, TT = np.meshgrid(XX, TT)
    plt.xlabel("X")
    plt.ylabel("T")
    axes.plot_surface(XX, TT, u1, cmap='rainbow')
    axes.set_title("u",loc='center')
    plt.title("u")
    plt.show()

    # sigma = 0.2
    # u2 = data_1d_noise(sigma=sigma)
    # for i in range(10):
    #     j = i*1000
    #     t = i/10*T
    #     plt.scatter(XX, u2[j], label='t={}'.format(t))
    # plt.title('sigma={}'.format(sigma))
    # plt.legend()
    # plt.show()