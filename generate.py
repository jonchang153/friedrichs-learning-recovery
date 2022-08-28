import numpy as np
import torch

def generate_1d_sol_grid(data):
    '''
    turn data of shape 20001 x 101 into an evenly spaced grid
    of 1001 time points and 101 spatial points
    '''

    data = torch.Tensor(data)

    temp=[]
    for i in range(1001):
        # data has shape 20001 x 101; temp is list of 1001 elements
        # temp is appended the spatial values of data for every 20th time point
        # from t = 0 up until the time t = 1000*20 = 20000
        # that is, every element of temp is a vector of 101 spatial values,
        # until unsqueeze(-1), which turns each value into a vector with 1 element,
        # so each element of temp is a 101 x 1 shaped matrix
        
        # pick out every 20th time point from indices 0, ..., 20000 (t=2)
        # and every spatial point from indices 0, ..., 100 (x=1)
        temp.append(data[20*i, :].unsqueeze(-1))

    # torch.cat concatenates temp along dim=0 so that the 1001 many 101 x 1 matrices
    # in temp become a 1001*101 x 1 = 101101 x 1 matrix
    grid = torch.cat(temp)

    # reshape so that you get 1001 many time points for every 20th time point
    # with 101 points of data for each time point
    grid = grid.reshape(1001, 101) 
    return grid


def generate_exact_sol_grid(func):
    '''
    grid generation function for the exact solution
    instead of data, func is a function that is called
    to generate the 1001 x 101 shaped grid
    '''

    T = 2
    X = np.pi

    dt = T / 1000
    dx = X / 100
    
    u = np.zeros((1001, 101))
    
    for j in range(1001):
        for k in range(101):
            t = j*dt
            x = k*dx
            u[j][k] = func(x,t)
            
    return torch.Tensor(u)


def generate_1d_domain():
    '''
    Generate domain [0,2] x [0,1] as evenly spaced grid with
    time points separated by 0.002 and spatial points separated by 0.01
    So, 1001 total time points and 101 total spatial points
    '''

    # add an extra dimension; now a 101 x 1 matrix
    x = torch.Tensor(np.linspace(0, 1, 101)).unsqueeze(-1)
    # x = torch.Tensor(np.linspace(0, np.pi, 101)).unsqueeze(-1)
    t = torch.Tensor(np.linspace(0, 2, 1001)).unsqueeze(-1)

    Z = []
    for i in range(len(x)):
        temp = torch.cat((t, torch.ones_like(t) * x[i][0]), dim=1)
        Z.append(temp)

    # Z has 101 elements, each of which is a 1001 x 2 shaped tensor
    # Each tensor runs through each of the 1001 time values in the first slot
    # and in the second slot has x value corresponding to the index
    # of the tensor k, x = k / 101

    # concatenates each value in list Z along dim=0; 101*1001 = 101101 x 2 shaped tensor
    # Thus, Z has pairs of points with every possible combination of 
    # t in [0, 0.002, 0.004, ... , 2] and
    # x in [0,  0.01,  0.02, ... , 1]
    Z = torch.cat(Z)

    return Z


def generate_simpson_weight_vector():
    '''
    For 1d numerical integration in time
    '''
    
    T = 2
    dt = T / 1000
    
    weight = torch.zeros(1001)
    for i in range(1, 1000):
        if i % 2 == 0:
            weight[i] = 2
        else:
            weight[i] = 4
    weight[0] = 1
    weight[1000] = 1
    weight = weight * dt / 3

    return weight

def generate_simpson_weight_vector2():
    '''
    For 1d numerical integration in space
    '''
    
    # X = np.pi
    X = 1
    dx = X / 100
    
    weight = torch.zeros(101)
    for i in range(1, 100):
        if i % 2 == 0:
            weight[i] = 2
        else:
            weight[i] = 4
    weight[0] = 1
    weight[100] = 1
    weight = weight * dx / 3

    return weight


def generate_simpson_weight_matrix():
    '''
    Generate coefficient matrix for simpson's rule for numerical integration
    See notes, for how 2-d simpson matrix is constructed
    '''
    
    T = 2
    # X = np.pi
    X = 1

    dt = T / 1000
    dx = X / 100

    weight = torch.zeros(1001, 101)
    for i in range(1, 1000):
        for j in range(1, 100):
            if i % 2 == 0:
                if j % 2 == 0:
                    weight[i][j] = 4
                else:
                    weight[i][j] = 8
            if i % 2 == 1:
                if j % 2 == 1:
                    weight[i][j] = 16
                else:
                    weight[i][j] = 8

    for j in range(1, 100):
        if j % 2 == 0:
            weight[0][j] = 2
            weight[-1][j] = 2
        else:
            weight[0][j] = 4
            weight[-1][j] = 4

    for i in range(1, 1000):
        if i % 2 == 0:
            weight[i][0] = 2
            weight[i][-1] = 2
        else:
            weight[i][0] = 4
            weight[i][-1] = 4

    weight[0][0] = 1
    weight[0][-1] = 1
    weight[-1][0] = 1
    weight[-1][-1] = 1
    weight = weight * dt * dx / 9

    return weight



def generate_1d_sol_grid_noise(sigma=0.1):

    u1 = data_1d_sol_noise(sigma=sigma)

    pass