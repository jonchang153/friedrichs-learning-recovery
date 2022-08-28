import os
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import pickle

import data
from data import *
from generate import *
from loss import *
from models import *


def train_1d_uniform_ls(param_dict):
    
    dtype = param_dict['dtype']

    lrG = param_dict['lrG']
    lrH = param_dict['lrH']
    
    flag_decay = param_dict['flag_decay']
    lrG_decay_rate = param_dict['lrG_decay_rate']
    lrH_decay_rate = param_dict['lrH_decay_rate']

    m = param_dict['m']

    n_epochs = param_dict['n_epochs']

    n_display_info = param_dict['n_display_info']
    n_graph_info = param_dict['n_graph_info']

    flag_noise = param_dict['flag_noise']
    sigma = param_dict['sigma']
    
    text = 'Training type: ls\n\n'
    text = text + 'Hyperparameters:\n'
    text = text + 'lrG: ' + str(lrG) + '\n'
    text = text + 'lrH: ' + str(lrH) + '\n'
    text = text + 'flag_decay: ' + str(flag_decay) + '\n'
    text = text + 'lrG_decay_rate: ' + str(lrG_decay_rate) + '\n'
    text = text + 'lrH_decay_rate: ' + str(lrH_decay_rate) + '\n'
    text = text + 'm: ' + str(m) + '\n'
    text = text + 'n_epochs: ' + str(n_epochs) + '\n\n'
    text = text + 'Function type:\n'
    if data.H_type == 1:
        text = text + 'H(x) = exp(-sin(2*pi*x))/20\n'
    elif data.H_type == 2:
        text = text + 'H(x) = x**2/10\n'
    if data.G_type == 1: 
        text = text + 'G(x) = cos(2*pi*x)\n\n'
    elif data.G_type == 2:
        text = text + 'G(x) = x**2\n\n'
    elif data.G_type == 3:
        text = text + 'G(x) = exp(sin(10*x))\n\n'
    print(text, end='')

#----------------------------------------------------------------

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set default data type for tensors
    torch.set_default_dtype(dtype)

    # spatial dimension of problem
    d = 1 

    # get timestamp for saving models
    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")

    # initialize errors
    best_H_error = float('inf')
    best_G_error = float('inf')
    
    # learning rate sequences
    lrG_seq = np.zeros(n_epochs)
    lrH_seq = np.zeros(n_epochs)

    for k in range(n_epochs):
        lrG_seq[k] = lrG * (1/10) ** (k/lrG_decay_rate)

    for k in range(n_epochs):
        lrH_seq[k] = lrH * (1/10) ** (k/lrH_decay_rate)

#----------------------------------------------------------------

    # width of each layer = m, besides input and output layers
    # input size = d, so d = 1 for H and G b/c functions of one variable x
    # and d = 2 for v b/c function of two variables x and t
    netH = net_H(d = d, m = m)
    netH = netH.to(device)
    netG = net_G(d = d, m = m)
    netG = netG.to(device)

    # initialize optimizers with respective parameters and learning rates
    optimG = torch.optim.Adam(netG.parameters(), lr=lrG)
    optimH = torch.optim.Adam(netH.parameters(), lr=lrH)

#----------------------------------------------------------------

    # generate data for u
    if flag_noise:
        u = generate_1d_sol_grid_noise(sigma=sigma).to(device)
    else:
        u = data_1d_u()
        u_t = generate_1d_sol_grid(data_1d_u_t(u)).to(device)
        u_x = generate_1d_sol_grid(data_1d_u_x(u)).to(device)
        u_xx = generate_1d_sol_grid(data_1d_u_xx(u)).to(device)
        u = generate_1d_sol_grid(u).to(device)

    # generate spatial grid
    x = torch.Tensor(np.linspace(0, 1, 101)).unsqueeze(-1)
    X = x.to(device)
    X.requires_grad = True

    # sequences to store info
    epoch_seq = []
    loss_seq = []
    H_error_seq = []
    G_error_seq = []

    # generate true data for H and G to compute error
    xx = np.linspace(0, 1, 101)
    h_true = [H(x) for x in xx]
    g_true = [G(x) for x in xx]
    h_true = torch.Tensor(h_true).to(device)
    g_true = torch.Tensor(g_true).to(device)

#----------------------------------------------------------------

    start_time = time.time()
    last_time = time.time()

    # run n_epochs many outer iterations (epochs)
    # for each outer iteration, run v_epochs many inner iterations to update v (maximize loss)
    for epoch in range(n_epochs):

        # set requires_grad to allow updates on H and G; freeze parameters in v
        for p in netH.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = True
            
        h, g = netH(X), netG(X)

        # positive so optimizer of H and G tries to minimize loss
        loss = train_1d_uniform_ls_loss(u, u_t, u_x, u_xx, h, g, X)

        optimG.zero_grad()
        optimH.zero_grad()
        loss.backward()
        optimG.step()
        optimH.step()

#----------------------------------------------------------------

        # Error calculation between parameterized H and G and true H and G 
        # using discrete relative L2 error
        H_error = torch.sqrt(torch.sum((h.squeeze() - h_true) ** 2) / torch.sum(h_true ** 2))
        G_error = torch.sqrt(torch.sum((g.squeeze() - g_true) ** 2) / torch.sum(g_true ** 2))

        # save errors, losses, and epoch info in corresponding sequences
        H_error_seq.append(H_error.cpu().detach().numpy())
        G_error_seq.append(G_error.cpu().detach().numpy())
        loss_seq.append(loss.item())
        epoch_seq.append(epoch)

        if epoch % n_display_info == 0 or epoch == n_epochs - 1:
            print(f'Info for epoch {epoch}:')
            print('        loss: %.4e' % (loss.item()))
            print('minimum loss: %.4e' % (min(loss_seq)))
            print('relative L2 error, H: %.4e, G: %.4e' % (H_error, G_error))
            print(' minimum L2 error, H: %.4e, G: %.4e' % (min(H_error_seq), min(G_error_seq)))
            print('time since start: %.3f s, '% (time.time() - start_time))
            print('time since last epoch: %.3f s\n'% (time.time() - last_time))

        last_time = time.time()
        
        # decrease learning rate
        if flag_decay:
            for param_group in optimG.param_groups:
                param_group['lr'] = lrG_seq[epoch]
            for param_group in optimH.param_groups:
                param_group['lr'] = lrH_seq[epoch]

        
#----------------------------------------------------------------
        
        # create directory for saving graphs and models
        path1 = f'./data_ls/graphs/{timestamp}/'
        path2 = f'./data_ls/models/{timestamp}/'

        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)

        if epoch % n_graph_info == 0 or epoch == n_epochs - 1:

            # graph netH against true H
            h = netH(X).cpu().detach().numpy()
            plt.plot(xx, h, label='netH')
            plt.plot(xx, [H(x) for x in xx], label='true H')
            plt.title('epoch: %d, relative L2 error for H: %.4e' % (epoch, H_error))
            plt.legend()
            plt.savefig(f'./data_ls/graphs/{timestamp}/{epoch}_H')
            # plt.show()
            plt.clf()

            # graph netG against true G
            g = netG(X).cpu().detach().numpy()
            plt.plot(xx, g, label='netG')
            plt.plot(xx, [G(x) for x in xx] , label='true G')
            plt.title('epoch: %d, relative L2 error for G: %.4e' % (epoch, G_error))
            plt.legend()
            plt.savefig(f'./data_ls/graphs/{timestamp}/{epoch}_G')
            # plt.show()
            plt.clf()
    
            # graph loss sequences
            plt.plot(epoch_seq, loss_seq, label='loss')
            plt.title(f'epoch {epoch} loss history')
            # ax = plt.gca()
            # ax.set_ylim(bottom=0)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.savefig(f'./data_ls/graphs/{timestamp}/{epoch}_loss')
            # plt.show()
            plt.clf()

            # graph error sequences
            plt.plot(epoch_seq, H_error_seq, label='H error')
            plt.plot(epoch_seq, G_error_seq, label='G error')
            plt.title(f'epoch {epoch} error history')
            # ax = plt.gca()
            # ax.set_ylim(bottom=0)
            plt.ylabel('relative L2 error')
            plt.xlabel('epoch')
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.savefig(f'./data_ls/graphs/{timestamp}/{epoch}_error')
            # plt.show()
            plt.clf()

        if epoch % n_graph_info == 0 or epoch == n_epochs - 1:
            state = {'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error}
            torch.save(state, f'./data_ls/models/{timestamp}/{epoch}.t7')
            
        # if H_error < best_H_error and G_error < best_G_error:
        #     state = {'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error, 'epoch': epoch}
        #     torch.save(state, f'./data_ls/models/{timestamp}/best.t7')
        #     best_H_error = H_error
        #     best_G_error = G_error
            
        if H_error < best_H_error:
            state = {'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error, 'epoch': epoch}
            torch.save(state, f'./data_ls/models/{timestamp}/bestH.t7')
            best_H_error = H_error
            
        if G_error < best_G_error:
            state = {'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error, 'epoch': epoch}
            torch.save(state, f'./data_ls/models/{timestamp}/bestG.t7')
            best_G_error = G_error
            
#----------------------------------------------------------------
    
    # save sequence data
    sequence = np.zeros((4, n_epochs))
    sequence[0,:] = epoch_seq
    sequence[1,:] = loss_seq
    sequence[2,:] = H_error_seq
    sequence[3,:] = G_error_seq
    with open('./data_ls/data/data_' + timestamp + '.data', 'wb') as f:
        pickle.dump(sequence, f)
        
    total_time = time.time() - start_time

    # print & save info after training
    text = text + 'Minimum loss: %.4e\n' % min(loss_seq)
    text = text + 'Minimum relative L2 error H: %.4e, G: %.4e\n' % (min(H_error_seq), min(G_error_seq))
    text = text + 'Training time: %.3f s, Avg per epoch: %.3f s' % (total_time, total_time / n_epochs)

    with open('./data_ls/logs/log_' + timestamp + '.txt', 'w') as f:   
        f.write(text)  

    print(text)
    
    # empty the GPU memory
    torch.cuda.empty_cache()