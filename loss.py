import torch

def gradient(y, x, grad_outputs=None):
    '''
    Compute gradient of y with respect to each input dimension in x
    x should be a batch of vector inputs and y should be a batch of scalar or vector outputs

    Use ones to indicate normal derivative calculation (see ex1-dis code)
    Output is wrapped in parentheses, so use [0] to get output

    Returns a vector the same size as x indicating the derivative of y at 
    the point of each vector input, with respect to each input dimension in the vector
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y).to(device)
    grad = torch.autograd.grad(y, x, grad_outputs = grad_outputs, create_graph=True, allow_unused=True)[0]
    return grad


def train_1d_uniform_fl_loss(u, v, h, g, Z, X, weight_m, weight_v):
    '''
    Loss function in complete weak form
    Needs u, v_t, v_x, v_xx
    '''

    # v are scalar outputs of shape 101101 x 1
    # h and g have shape 101 x 1 because X is unsqueezed

    # returns vector of ∂v/∂t and ∂v/∂x at each input in Z
    v_tx = gradient(v, Z)

    # pick out the first derivatives wrt t
    v_t = v_tx[:, 0].reshape(101, 1001).T

    # pick out the first derivatives wrt x
    v_x = v_tx[:, 1].reshape(101, 1001).T

    # calculate ∂/∂t (∂v/∂x) and ∂/∂x (∂v/∂x) at each input in Z
    # then, pick out only the second derivatives wrt x, ∂^2 v / ∂x^2
    v_xx = gradient(v_x, Z)[:, 1].reshape(101, 1001).T

    # remove last dimension so become one 101101 sized vector;
    # reshape to matrix of shape 1001 x 101
    v = v.squeeze().reshape(101, 1001).T

    # all of the above outputs are shape 101101 x 1, with
    # a scalar output for each of the possible combinations
    # of 101 spatial and 1001 time points.
    # in particular, Z had pairs ordered such that each 1001
    # pairs ran through each of the time points for one spatial point (shape 101101 x 2)

    # then, reshaping outputs to 101 x 1001 gives 101 vectors of length 1001
    # where each vector gives the scalar outputs at the 1001 time points
    # for the single spatial point corresponding to that vector.
    # Then we apply .T (transpose) to reshape to 1001 x 101, where now
    # we have 1001 vectors corresponding to each time point, filled with
    # 101 scalars for each spatial point

    h_x = gradient(h, X) # X gives scalar inputs, so output is just dh/dx

    # v_t, v_x, v_xx, v have shape 1001 x 101
    # h_x, h, g         have shape 101 x 1
    # So, apply transpose; then shape 1 x 101
    # So the values of h_x, h, g give scalar outputs for each spatial point
    # and multiplying 1001 x 101 and 1 x 101 multiplies the 101 values
    # pointwise for each of the 1001 vectors of length 101
    inte = -u * (v_t + v_x * (h_x.T) + v_xx * (h.T) + v * (g.T)) # inte is shape 101 x 1001

    # multiply pointwise by simpson matrix and sum all entries to get integral value
    integral_1 = torch.sum(inte*weight_m)

    # expression on boundary ∂Ω
    lhs = u[:,-1] * h[-1] * v_x[:, -1]
    rhs = u[:,0] * h[0]* v_x[:,0]
    integral_2 = torch.sum(weight_v * (lhs - rhs))

    integral = integral_1 + integral_2

    # normalization factor
    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)+torch.sum(v_t*v_t*weight_m)+torch.sum(v_x*v_x*weight_m)) # H1 norm
    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm

    # finalized loss along with normalization
    # additional mean(v*v) term is very big for large v,
    # so penalize (increase loss)
    loss = torch.abs(integral) / l2_v #+ 0.000001*torch.mean(v*v)

    return loss

    # case where v_x needs to vanish on the boundary; add derivative term to loss
    # c = 1
    # loss = torch.abs(integral) / l2_v + c * (v_x[:,-1] + v_x[:,0])

    # other normalization factors:
    # l2_v = torch.sqrt(torch.mean(v*v)+torch.mean(v_tx*v_tx))
    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm
    

def riemann(u, v, h, g, Z, X):
    
    v_tx = gradient(v, Z)

    v_t = v_tx[:, 0].reshape(101, 1001).T

    v_x = v_tx[:, 1].reshape(101, 1001).T

    v_xx = gradient(v_x, Z)[:, 1].reshape(101, 1001).T

    v = v.squeeze().reshape(101, 1001).T

    h_x = gradient(h, X) 




    inte = -u * (v_t + v_x * (h_x.T) + v_xx * (h.T) + v * (g.T)) 
    
    inte1, inte2, inte3, inte4 = inte, inte, inte, inte

    inte1[1000, :] = 0
    inte1[:, 100] = 0
    inte2[0, :] = 0
    inte2[:, 100] = 0
    inte3[1000, :] = 0
    inte3[:, 0] = 0
    inte4[0, :] = 0
    inte4[:, 0] = 0

    integral_1 = (torch.sum(inte1)+torch.sum(inte2)+torch.sum(inte3)+torch.sum(inte4)) / 4 * 2 / 1000 / 100


    lhs = h[-1] * u[:,-1] * v_x[:, -1]
    rhs = h[0] * u[:,0] * v_x[:,0]
    inte = lhs - rhs 
    inte5, inte6 = inte, inte
    inte5[1000] = 0
    inte6[0] = 0

    integral_2 = (torch.sum(inte5)+torch.sum(inte6)) / 2 * 2 / 1000




    integral = integral_1 + integral_2

    l2_v = torch.sqrt(torch.sum(v*v) * 2 / 1000 / 100) # L2 norm

    loss = torch.abs(integral) / l2_v #+ 0.000001*torch.mean(v*v)

    return loss


def loss_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    '''
    Loss function without any integration by parts
    Needs u, u_t, u_x, u_xx, v
    '''

    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - (h_x.T) * u_x - (h.T) * u_xx - (g.T) * u) * v
    integral = torch.sum(weight_m * inte)

    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm

    loss = torch.abs(integral) / l2_v
    return loss

def loss_3(u, u_x, u_xx, v, h, g, Z, X, weight_m):
    '''
    Loss function with one ibp of u wrt t
    Needs u, u_x, u_xx, v, v_t
    '''

    v_tx = gradient(v, Z)
    v_t = v_tx[:, 0].reshape(101, 1001).T
    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = u * v_t + v * ((h_x.T) * u_x + (h.T) * u_xx + (g.T) * u)
    integral = torch.sum(weight_m * inte)

    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)+torch.sum(v_t*v_t*weight_m)) # H1 norm
    l2_v = torch.sqrt(torch.sum(v*v*weight_m))

    loss = torch.abs(integral) / l2_v
    return loss
    
def loss_4(u, u_t, u_x, v, h, g, Z, X, weight_m):
    '''
    Loss function with one ibp of u wrt x
    Needs u, u_t, u_x, v, v_x
    '''

    v_tx = gradient(v, Z)
    v_x = v_tx[:, 1].reshape(101, 1001).T
    v = v.squeeze().reshape(101, 1001).T

    inte = u_t * v + (h.T) * v_x * u_x - (g.T) * u * v
    integral = torch.sum(weight_m * inte)
    
    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)+torch.sum(v_x*v_x*weight_m)) # H1 norm
    l2_v = torch.sqrt(torch.sum(v*v*weight_m))

    loss = torch.abs(integral) / l2_v
    return loss

def loss_5(u, u_x, v, h, g, Z, X, weight_m):
    '''
    Loss function with one ibp of u wrt t and one ibp of u wrt x
    Needs u, u_x, v, v_t, v_x
    '''

    v_tx = gradient(v, Z)
    v_t = v_tx[:, 0].reshape(101, 1001).T
    v_x = v_tx[:, 1].reshape(101, 1001).T
    v = v.squeeze().reshape(101, 1001).T

    inte = -u * v_t + (h.T) * v_x * u_x - (g.T) * u * v
    integral = torch.sum(weight_m * inte)

    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)+torch.sum(v_t*v_t*weight_m)+torch.sum(v_x*v_x*weight_m)) # H1 norm
    l2_v = torch.sqrt(torch.sum(v*v*weight_m))

    loss = torch.abs(integral) / l2_v
    return loss

def loss_6(u, u_t, v, h, g, Z, X, weight_m, weight_v):
    '''
    Loss function with two ibp of u wrt x
    Needs u, u_t, v, v_x, v_xx
    '''

    v_tx = gradient(v, Z)
    v_t = v_tx[:, 0].reshape(101, 1001).T
    v_x = v_tx[:, 1].reshape(101, 1001).T
    v_xx = gradient(v_x, Z)[:, 1].reshape(101, 1001).T
    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = u_t * v - u * (v_x * (h_x.T) + v_xx * (h.T) + v * (g.T))
    integral_1 = torch.sum(inte*weight_m)

    lhs = h[-1] * u[:,-1] * v_x[:, -1]
    rhs = h[0] * u[:,0] * v_x[:,0]
    integral_2 = torch.sum(weight_v * (lhs - rhs))

    integral = integral_1 + integral_2

    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm

    loss = torch.abs(integral) / l2_v
    return loss


def train_1d_uniform_ls_loss(u, u_t, u_x, u_xx, h, g, X):
    
    h_x = gradient(h, X)

    loss = torch.abs(u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T))
    loss = torch.sum(loss * loss) / 101 / 1001

    return loss





def test1(u, u_t, u_x, u_xx, h, g, X):
    # strong form
    
    h_x = gradient(h, X)

    loss = torch.abs(u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T))
    loss = torch.sum(loss * loss) / 101 / 1001

    return loss

def test2(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # L2 inner product with v

    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    integral = torch.sum(weight_m * inte)

    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm; constant value of sqrt(2pi)

    loss = torch.abs(integral) / l2_v
    return loss

def test2_1(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # L2 inner product with v, abs value on inside squared

    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    integral = torch.sum(weight_m * torch.square(torch.abs(inte)))

    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm; constant value of sqrt(2pi)

    loss = integral / l2_v
    return loss

def test2_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # L2 inner product with v, square outside

    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    integral = torch.sum(weight_m * inte)

    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm; constant value of sqrt(2pi)

    loss = torch.square(torch.abs(integral)) / l2_v
    return loss

def test3(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # multiply by v and avg loss
    
    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    loss = torch.sum(torch.abs(inte)) / 1001 / 101
    
    return loss

def test4(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # multiply by v and avg square loss
    
    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    loss = torch.sum(inte * inte) / 1001 / 101
    
    return loss

def test5(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # integrate only
    
    h_x = gradient(h, X)

    inte = u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)
    
    integral = torch.sum(weight_m * inte)
    loss = torch.abs(integral)
    
    return loss

def test6(u, u_t, u_x, u_xx, v, h, g, X, weight_m):
    # L2 norm with v
    
    v = v.squeeze().reshape(101, 1001).T
    h_x = gradient(h, X)

    inte = (u_t - u_x * (h_x.T) - u_xx * (h.T) - u * (g.T)) * v
    
    integral = torch.sum(weight_m * inte * inte)
    loss = torch.sqrt(integral)
    
    return loss





def train_1d_uniform_fl_loss2(u, u_x, v, h, g, Z, X, weight_m, weight_v, weight_v2):

    v_tx = gradient(v, Z)
    v_t = v_tx[:, 0].reshape(101, 1001).T
    v_x = v_tx[:, 1].reshape(101, 1001).T
    v_xx = gradient(v_x, Z)[:, 1].reshape(101, 1001).T
    v = v.squeeze().reshape(101, 1001).T

    h_x = gradient(h, X)

    inte = -u * (v_t + v_x * (h_x.T) + v_xx * (h.T) + v * (g.T))

    integral_1 = torch.sum(inte*weight_m)

    lhs = u[:,-1]*h[-1]*v_x[:, -1] - v[:,-1]*h[-1]*u_x[:,-1]
    rhs = u[:,0]*h[0]*v_x[:,0] - v[:,0]*h[0]*u_x[:,0]
    integral_2 = torch.sum(weight_v * (lhs - rhs))

    lhs2 = u[-1,:]*v[-1,:]
    rhs2 = u[0,:]*v[0,:]
    integral_3 = torch.sum(weight_v2 * (lhs2 - rhs2))

    integral = integral_1 + integral_2 + integral_3

    # l2_v = torch.sqrt(torch.sum(v*v*weight_m)+torch.sum(v_t*v_t*weight_m)+torch.sum(v_x*v_x*weight_m)) # H1 norm
    l2_v = torch.sqrt(torch.sum(v*v*weight_m)) # L2 norm

    loss = torch.abs(integral) / l2_v

    return loss