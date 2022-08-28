# When switching to domain of [0, π] example:
# Change X and xx generations in train function
# Change data_1d_u functions in data file and ivc (if using numerical solution)
# Change generate_1d_domain and simpson weight functions in generate file
# Change a function in neural nets

# Check gpu type: python -m torch.utils.collect_env

from train_1d_uniform_fl import *

param_dict_fl = {
    'dtype': torch.float64, # torch.float64 (double) or torch.float32 (single)

    'lrG': 0.0003, # initial learning rate for G
    'lrH': 0.0003, # initial learning rate for H
    'lrv': 0.003, # initial learning rate for v

    'flag_decay': True,
    'lrG_decay_rate': 20000, # learning rate decay for G
    'lrH_decay_rate': 20000, # learning rate decay for H
    'lrv_decay_rate': 20000, # learning rate decay for v

    'ms': 50, # width of each solution NN
    'mt': 50, # width of each test NN

    'n_epochs': 50000, # number of outer iterations
    'v_epochs': 10, # number of inner iterations for v
    'HG_epochs': 1, # number of inner iterations for H and G

    'n_display_info': 10, # number of outer iterations before displaying info
    'n_graph_info': 1000, # number of outer iterations before graphing info

    'flag_noise': False, # if use noisy data for u
    'sigma': 0.00, # if noisy data, value of sigma to use
}

# param_dict_ls = {
#     'dtype': torch.float64, # torch.float64 (double) or torch.float32 (single)

#     'lrG': 0.001, # initial learning rate for G
#     'lrH': 0.001, # initial learning rate for H
#     'lrv': 0.003, # initial learning rate for v

#     'flag_decay': True,
#     'lrG_decay_rate': 20000, # learning rate decay for G
#     'lrH_decay_rate': 20000, # learning rate decay for H
#     'lrv_decay_rate': 20000, # learning rate decay for v

#     'ms': 50, # width of each solution NN
#     'mt': 50, # width of each test NN

#     'n_epochs': 50000, # number of outer iterations
#     'v_epochs': 0, # number of inner iterations for v
#     'HG_epochs': 1, # number of inner iterations for H and G

#     'n_display_info': 100, # number of outer iterations before displaying info
#     'n_graph_info': 2000, # number of outer iterations before graphing info

#     'flag_noise': False, # if use noisy data for u
#     'sigma': 0.00, # if noisy data, value of sigma to use
# }

# data.H_type = 1
# data.G_type = 1
# train_1d_uniform_fl(param_dict_ls, loss = 2, nntype = 2)
# train_1d_uniform_fl(param_dict_fl, loss = 1, nntype = 2)
# data.H_type = 2
# data.G_type = 2
# train_1d_uniform_fl(param_dict_ls, loss = 2, nntype = 2)
# train_1d_uniform_fl(param_dict_fl, loss = 1, nntype = 2)

# data.H_type = 1
# data.G_type = 1
# train_1d_uniform_fl(param_dict_fl, loss = 1, nntype = 2)
# data.H_type = 2
# data.G_type = 2
# train_1d_uniform_fl(param_dict_fl, loss = 1, nntype = 2)




from train_1d_uniform_ls import *

param_dict_ls = {
    'dtype': torch.float64, # torch.float64 (double) or torch.float32 (single)

    'lrG': 0.001, # learning rate for G
    'lrH': 0.001, # learning rate for H
    
    'flag_decay': True,
    'lrG_decay_rate': 20000, # learning rate decay for G
    'lrH_decay_rate': 20000, # learning rate decay for H

    'm': 50, # width of each NN

    'n_epochs': 100000, # number of outer iterations

    'n_display_info': 100, # number of outer iterations before displaying info
    'n_graph_info': 2000, # number of outer iterations before graphing info

    'flag_noise': False, # if use noisy data for u
    'sigma': 0.00, # if noisy data, value of sigma to use
}
param_dict_ls2 = {
    'dtype': torch.float64, # torch.float64 (double) or torch.float32 (single)

    'lrG': 0.001, # learning rate for G
    'lrH': 0.001, # learning rate for H
    
    'flag_decay': False,
    'lrG_decay_rate': 20000, # learning rate decay for G
    'lrH_decay_rate': 20000, # learning rate decay for H

    'm': 50, # width of each NN

    'n_epochs': 100000, # number of outer iterations

    'n_display_info': 100, # number of outer iterations before displaying info
    'n_graph_info': 2000, # number of outer iterations before graphing info

    'flag_noise': False, # if use noisy data for u
    'sigma': 0.00, # if noisy data, value of sigma to use
}
data.H_type = 1
data.G_type = 1
try:
    train_1d_uniform_ls(param_dict_ls2)
except:
    pass
try:
    train_1d_uniform_ls(param_dict_ls)
except:
    pass

data.H_type = 2
data.G_type = 2
try:
    train_1d_uniform_ls(param_dict_ls2)
except:
    pass
try:
    train_1d_uniform_ls(param_dict_ls)
except:
    pass

train_1d_uniform_fl(param_dict_fl, loss = 1, nntype = 2)