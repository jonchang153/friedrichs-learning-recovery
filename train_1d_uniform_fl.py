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


def train_1d_uniform_fl(param_dict, loss, nntype):

    dtype = param_dict['dtype']

    lrG = param_dict['lrG']
    lrH = param_dict['lrH']
    lrv = param_dict['lrv']

    flag_decay = param_dict['flag_decay']
    lrG_decay_rate = param_dict['lrG_decay_rate']
    lrH_decay_rate = param_dict['lrH_decay_rate']
    lrv_decay_rate = param_dict['lrv_decay_rate']

    ms = param_dict['ms']
    mt = param_dict['mt']

    n_epochs = param_dict['n_epochs']
    v_epochs = param_dict['v_epochs']
    HG_epochs = param_dict['HG_epochs']

    n_display_info = param_dict['n_display_info']
    n_graph_info = param_dict['n_graph_info']

    flag_noise = param_dict['flag_noise']
    sigma = param_dict['sigma']
    
    text = 'Training type: fl\n'
    text = text + 'Loss type: ' + str(loss) + '\n'
    text = text + 'Net type: ' + str(nntype) + '\n\n'
    text = text + 'Hyperparameters:\n'
    text = text + 'lrG: ' + str(lrG) + '\n'
    text = text + 'lrH: ' + str(lrH) + '\n'
    text = text + 'lrv: ' + str(lrv) + '\n'
    text = text + 'flag_decay: ' + str(flag_decay) + '\n'
    text = text + 'lrG_decay_rate: ' + str(lrG_decay_rate) + '\n'
    text = text + 'lrH_decay_rate: ' + str(lrH_decay_rate) + '\n'
    text = text + 'lrv_decay_rate: ' + str(lrv_decay_rate) + '\n'
    text = text + 'ms: ' + str(ms) + '\n'
    text = text + 'mt: ' + str(mt) + '\n'
    text = text + 'n_epochs: ' + str(n_epochs) + '\n'
    text = text + 'v_epochs: ' + str(v_epochs) + '\n'
    text = text + 'HG_epochs: ' + str(HG_epochs) + '\n\n'
    text = text + 'Function type:\n'
    if data.H_type == 1:
        text = text + 'H(x) = exp(-sin(2*pi*x))/20\n'
    elif data.H_type == 2:
        text = text + 'H(x) = x**2/10\n'
    elif data.H_type == 3:
        text = text + 'H(x) = 0\n'
    elif data.H_type == 4:
        text = text + 'H(x) = 2\n'
    if data.G_type == 1:
        text = text + 'G(x) = cos(2*pi*x)\n\n'
    elif data.G_type == 2:
        text = text + 'G(x) = x**2\n\n'
    elif data.G_type == 3:
        text = text + 'G(x) = -1\n\n'
    elif data.G_type == 4:
        text = text + 'G(x) = 0.5\n\n'
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
    lrv_seq = np.zeros(n_epochs)

    for k in range(n_epochs):
        lrG_seq[k] = lrG * (1/10) ** (k/lrG_decay_rate)

    for k in range(n_epochs):
        lrH_seq[k] = lrH * (1/10) ** (k/lrH_decay_rate)

    for k in range(n_epochs):
        lrv_seq[k] = lrv * (1/10) ** (k/lrv_decay_rate)

#----------------------------------------------------------------

    # width of each layer = m, besides input and output layers
    # input size = d, so d = 1 for H and G b/c functions of one variable x
    # and d = 2 for v b/c function of two variables x and t
    
    if nntype == 1:
        netH = net_H(d = d, m = ms)
        netH = netH.to(device)
        netG = net_G(d = d, m = ms)
        netG = netG.to(device)
        netv = net_v(d = d+1, m = mt)
        netv = netv.to(device)
        
    elif nntype == 2:
        # netH = ReLUResNet(d = d, m = ms, degree = 2, test = False)
        netH = TanhResNet(d = d, m = ms, test = False)
        netH = netH.to(device)
        # netG = ReLUResNet(d = d, m = ms, degree = 1, test = False)
        netG = TanhResNet(d = d, m = ms, test = False)
        netG = netG.to(device)
        # netv = ReLUResNet(d = d+1, m = mt, degree = 3, test = True)
        # netv = TanhResNet(d = d+1, m = mt, test = True)
        # netv = TanhResNet2(d = d+1, m = mt, test = True) # for testing, gives fixed v = a(x,t)
        netv = TanhResNet(d = d+1, m = mt, test = True)
        netv = netv.to(device)

    elif nntype == 3:
        netH = TanhResNet(d = d, m = ms, test = False)
        netH = netH.to(device)
        netG = TanhResNet(d = d, m = ms, test = False)
        netG = netG.to(device)
        netv = TanhResNet3(d = d+1, m = mt, test = True) # for testing, gives fixed v = 1
        netv = netv.to(device)

    elif nntype == 4:
        netH = TanhResNet(d = d, m = ms, test = False)
        netH = netH.to(device)
        netG = TanhResNet(d = d, m = ms, test = False)
        netG = netG.to(device)
        netv = TanhResNet4(d = d+1, m = mt, test = True) # for testing, gives v not in H_0^1
        netv = netv.to(device)


    # initialize optimizers with respective parameters and learning rates
    optimG = torch.optim.Adam(netG.parameters(), lr=lrG)
    optimH = torch.optim.Adam(netH.parameters(), lr=lrH)
    optimv = torch.optim.Adam(netv.parameters(), lr=lrv)

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

        # for test with exact soln
        # u = generate_exact_sol_grid(exact_u).to(device)
        # u_t = generate_exact_sol_grid(exact_u_t).to(device)
        # u_x = generate_exact_sol_grid(exact_u_x).to(device)
        # u_xx = generate_exact_sol_grid(exact_u_xx).to(device)

    # generate domain grid
    Z = generate_1d_domain().to(device)
    Z.requires_grad = True

    # generate spatial grid
    x = torch.Tensor(np.linspace(0, 1, 101)).unsqueeze(-1)
    # x = torch.Tensor(np.linspace(0, np.pi, 101)).unsqueeze(-1)
    X = x.to(device)
    X.requires_grad = True
    
    # generate simpson weight
    weight_m = generate_simpson_weight_matrix().to(device)
    weight_v = generate_simpson_weight_vector().to(device)
    weight_v2 = generate_simpson_weight_vector2().to(device)

    # sequences to store info
    epoch_seq = []
    loss1_seq = []
    loss2_seq = []
    H_error_seq = []
    G_error_seq = []
    
    # generate true data for H and G to compute error
    xx = np.linspace(0, 1, 101)
    # xx = np.linspace(0, np.pi, 101)
    h_true = [H(x) for x in xx]
    g_true = [G(x) for x in xx]
    h_true = torch.Tensor(h_true).to(device)
    g_true = torch.Tensor(g_true).to(device)

#----------------------------------------------------------------

    start_time = time.time()
    last_time = time.time()
    
    # run n_epochs many outer iterations (epochs)
    # for each outer iteration, run v_epochs many inner iterations to update v (maximize loss)
    # for each outer iteration, run HG_epochs many inner iterations to update H and G (minimize loss)
    for epoch in range(n_epochs):

        # set requires_grad to allow updates on v; freeze parameters in H and G
        for p in netv.parameters():
            p.requires_grad = True
        for p in netH.parameters():
            p.requires_grad = False
        for p in netG.parameters():
            p.requires_grad = False

        # inner iterations for v
        for v_epoch in range(v_epochs):

            v, h, g = netv(Z), netH(X), netG(X)

            # negative so optimizer of v tries to maximize loss
            if loss == 1:
                loss1 = -train_1d_uniform_fl_loss(u, v, h, g, Z, X, weight_m, weight_v)
                # loss1 = -train_1d_uniform_fl_loss2(u, u_x, v, h, g, Z, X, weight_m, weight_v, weight_v2)
                # loss1 = -test2_1(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 2:
                # loss1 = -loss_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                # loss1 = -test2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                # loss1 = -test2_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                loss1 = -train_1d_uniform_fl_loss(u, v, h, g, Z, X, weight_m, weight_v)
            elif loss == 3:
                # loss1 = -loss_3(u, u_x, u_xx, v, h, g, Z, X, weight_m)
                loss1 = -test3(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 4:
                # loss1 = -loss_4(u, u_t, u_x, v, h, g, Z, X, weight_m)
                loss1 = -test4(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 5:
                loss1 = -loss_5(u, u_x, v, h, g, Z, X, weight_m)
            elif loss == 6:
                loss1 = -loss_6(u, u_t, v, h, g, Z, X, weight_m, weight_v)

            optimv.zero_grad()
            loss1.backward()
            optimv.step()
            
        if v_epochs == 0:
            loss1 = -torch.zeros(1)

        # set requires_grad to allow updates on H and G; freeze parameters in v
        for p in netH.parameters():
            p.requires_grad = True
        for p in netG.parameters():
            p.requires_grad = True
        for p in netv.parameters():
            p.requires_grad = False

        # inner iterations for H and G
        for HG_epoch in range(HG_epochs):
            
            v, h, g = netv(Z), netH(X), netG(X)

            # positive so optimizer of H and G tries to minimize loss
            if loss == 1:
                loss2 = train_1d_uniform_fl_loss(u, v, h, g, Z, X, weight_m, weight_v)
                # loss2 = train_1d_uniform_fl_loss2(u, u_x, v, h, g, Z, X, weight_m, weight_v, weight_v2)
                # loss2 = train_1d_uniform_ls_loss(u, u_t, u_x, u_xx, h, g, X)
                # loss2 = test1(u, u_t, u_x, u_xx, h, g, X)
                # loss2 = test2_1(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 2:
                # loss2 = loss_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                # loss2 = test2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                # loss2 = test2_2(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
                # loss2 = train_1d_uniform_fl_loss(u, v, h, g, Z, X, weight_m, weight_v)
                loss2 = train_1d_uniform_ls_loss(u, u_t, u_x, u_xx, h, g, X)
            elif loss == 3:
                # loss2 = loss_3(u, u_x, u_xx, v, h, g, Z, X, weight_m)
                loss2 = test3(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 4:
                # loss2 = loss_4(u, u_t, u_x, v, h, g, Z, X, weight_m)
                loss2 = test4(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 5:
                # loss2 = loss_5(u, u_x, v, h, g, Z, X, weight_m)
                loss2 = test5(u, u_t, u_x, u_xx, v, h, g, X, weight_m)
            elif loss == 6:
                # loss2 = loss_6(u, u_t, v, h, g, Z, X, weight_m, weight_v)
                loss2 = test6(u, u_t, u_x, u_xx, v, h, g, X, weight_m)

            optimG.zero_grad()
            optimH.zero_grad()
            loss2.backward()
            optimG.step()
            optimH.step()

#----------------------------------------------------------------

        # Error calculation between parameterized H and G and true H and G 
        # using discrete relative L2 error
        if data.H_type == 3:
            H_error = torch.sqrt(torch.sum((h.squeeze() - h_true) ** 2))
        else:
            H_error = torch.sqrt(torch.sum((h.squeeze() - h_true) ** 2) / torch.sum(h_true ** 2))
        G_error = torch.sqrt(torch.sum((g.squeeze() - g_true) ** 2) / torch.sum(g_true ** 2))

        # save errors, losses, and epoch info in corresponding sequences
        H_error_seq.append(H_error.cpu().detach().numpy())
        G_error_seq.append(G_error.cpu().detach().numpy())
        loss1_seq.append(-loss1.item())
        loss2_seq.append(loss2.item())
        epoch_seq.append(epoch)

        if epoch % n_display_info == 0 or epoch == n_epochs - 1:
            print(f'Info for epoch {epoch}:')
            print('        loss1: %.4e, loss2: %.4e' % (-loss1.item(), loss2.item()))
            print('minimum loss1: %.4e, loss2: %.4e' % (min(loss1_seq), min(loss2_seq)))
            print('relative L2 error, H: %.4e, G: %.4e' % (H_error, G_error))
            print(' minimum L2 error, H: %.4e, G: %.4e' % (min(H_error_seq), min(G_error_seq)))
            print('time since start: %.3f s, '% (time.time() - start_time), end='')
            print('time since last epoch: %.3f s'% (time.time() - last_time))
            print('v_mean: %.4e, v_max: %.4e\n' % (torch.mean(v*v).item(), torch.max(v*v).item()))

        last_time = time.time()
        
        # decrease learning rate
        if flag_decay:
            for param_group in optimG.param_groups:
                param_group['lr'] = lrG_seq[epoch]
            for param_group in optimH.param_groups:
                param_group['lr'] = lrH_seq[epoch]
            for param_group in optimv.param_groups:
                param_group['lr'] = lrv_seq[epoch]
                
#----------------------------------------------------------------
        
        # create directory for saving graphs
        path1 = f'./data_fl/graphs/{timestamp}/'
        path2 = f'./data_fl/models/{timestamp}/'
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
            plt.grid()
            plt.savefig(f'./data_fl/graphs/{timestamp}/{epoch}_H')
            # plt.show()
            plt.clf()

            # graph netG against true G
            g = netG(X).cpu().detach().numpy()
            plt.plot(xx, g, label='netG')
            plt.plot(xx, [G(x) for x in xx] , label='true G')
            plt.title('epoch: %d, relative L2 error for G: %.4e' % (epoch, G_error))
            plt.legend()
            plt.grid()
            plt.savefig(f'./data_fl/graphs/{timestamp}/{epoch}_G')
            # plt.show()
            plt.clf()

            # graph loss sequences
            plt.plot(epoch_seq, loss1_seq, label='loss1')
            plt.plot(epoch_seq, loss2_seq, label='loss2')
            plt.title(f'epoch {epoch} loss history')
            # ax = plt.gca()
            # ax.set_ylim(bottom=0)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.savefig(f'./data_fl/graphs/{timestamp}/{epoch}_loss')
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
            plt.savefig(f'./data_fl/graphs/{timestamp}/{epoch}_error')
            # plt.show()
            plt.clf()

        # if epoch % n_graph_info == 0 or epoch == n_epochs - 1:
        #     state = {'netv': netv, 'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error}
        #     torch.save(state, f'./data_fl/models/{timestamp}/{epoch}.t7')

        if H_error < best_H_error:
            state = {'netv': netv, 'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error, 'epoch': epoch}
            torch.save(state, f'./data_fl/models/{timestamp}/bestH.t7')
            best_H_error = H_error
            
        if G_error < best_G_error:
            state = {'netv': netv, 'netH': netH, 'netG': netG, 'H_error': H_error, 'G_error': G_error, 'epoch': epoch}
            torch.save(state, f'./data_fl/models/{timestamp}/bestG.t7')
            best_G_error = G_error

#----------------------------------------------------------------
    
    # save sequence data
    sequence = np.zeros((5, n_epochs))
    sequence[0,:] = epoch_seq
    sequence[1,:] = loss1_seq
    sequence[2,:] = loss2_seq
    sequence[3,:] = H_error_seq
    sequence[4,:] = G_error_seq
    with open('./data_fl/data/data_' + timestamp + '.data', 'wb') as f:
        pickle.dump(sequence, f)
        
    total_time = time.time() - start_time

    # print & save info after training
    text = text + 'Minimum loss1: %.4e, loss2: %.4e\n' % (min(loss1_seq), min(loss2_seq))
    text = text + 'Minimum relative L2 error H: %.4e, G: %.4e\n' % (min(H_error_seq), min(G_error_seq))
    text = text + 'Training time: %.3f s, Avg per epoch: %.3f s' % (total_time, total_time / n_epochs)

    with open('./data_fl/logs/log_' + timestamp + '.txt', 'w') as f:   
        f.write(text)

    print(text)
    
    # empty the GPU memory
    torch.cuda.empty_cache()