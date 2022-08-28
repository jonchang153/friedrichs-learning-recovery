import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class net_v(nn.Module):
    def __init__(self, d, m, ngpu=1, nex=4096):
        super(net_v, self).__init__()
        self.ngpu = ngpu

        self.Relu = nn.ReLU()
        self.Linear1 = nn.Linear(d, m)
        self.Linear2 = nn.Linear(m, m)
        self.Linear3 = nn.Linear(m, m)
        self.Linear4 = nn.Linear(m, 1)

    def forward(self, x):
        # dim(x) includes time t (first dimension)
        # Relu^3 used to allow second differentiability in v
        # +out to utilize ResNet structure
        out = self.Linear1(x)
        out = self.Relu(out)**3+out 
        out = self.Linear2(out)
        out = self.Relu(out)**3
        out = self.Linear3(out)
        out = self.Relu(out)**3+out
        output = self.Linear4(out)

        # x[:,1] represents all the spatial inputs; 0 whenever spatial input is on boundary (0 or 1)
        # x[:,0] represents all the temporal inputs; 0 whenever temporal input is on boundary (0 or 2)
        a = (1-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0]
        # a(x,t) is a function such that v(x,t) * a(x,t) = 0 on the boundaries (arbitrarily chosen)

        # need to squeeze bc original inputs are size d,
        # but reduced to size 1, so can remove last dimension
        # and result in one dimension tensor being 
        # number of original samples, so this multiplication is possible
        output = output.squeeze()*a

        # then returned matrix still needs each value in its own vector
        return output.unsqueeze(-1)

class net_H(nn.Module):
    def __init__(self, d, m, ngpu=1, nex=4096):
        super(net_H, self).__init__()
        self.ngpu = ngpu

        self.Linear1 = nn.Linear(d, m)
        self.Linear2 = nn.Linear(m, m)
        self.Linear3 = nn.Linear(m, m)
        self.Linear4 = nn.Linear(m, 1)
        self.Relu = nn.ReLU()

    def forward(self, input):
        # output = self.main(input)
        # Relu^2 used to allow first derivative in H
        out = self.Linear1(input)
        out = self.Relu(out)**2
        out = self.Linear2(out)
        out = self.Relu(out)**2
        out = self.Linear3(out)
        out = self.Relu(out)**2
        output = self.Linear4(out)

        return output

class net_G(nn.Module):
    def __init__(self, d, m, ngpu=1, nex=4096):
        super(net_G, self).__init__()
        self.ngpu = ngpu

        self.Linear1 = nn.Linear(d,m)
        self.Linear2 = nn.Linear(m, m)
        self.Linear3 = nn.Linear(m,m)
        self.Linear4 = nn.Linear(m,1)
        self.Relu = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, input):
        # output = self.main(input)
        out = self.Linear1(input)
        out = self.Tanh(out)
        out = self.Linear2(out)
        out = self.Tanh(out) + out
        out = self.Linear3(out)
        out = self.Tanh(out)
        output = self.Linear4(out)

        return output

    
#----------------------------------------------------------------


class TanhResNet(nn.Module):
    def __init__(self, d, m, test):
        super(TanhResNet, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.outlayer = nn.Linear(m, 1, bias = False)
        
        self.d = d
        self.test = test
        self.Tanh = nn.Tanh()
        
        Ix = torch.zeros([d,m]).to(device)
        for i in range(d):
            Ix[i,i] = 1
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.Tanh(y)
        # y = self.fc2(y)
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = self.Tanh(y)
        # y = self.fc4(y)       
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = self.Tanh(y)
        # y = self.fc6(y)    
        # y = self.Tanh(y)
        y = y+s
                
        y = self.outlayer(y)
        
        if self.test:
            a = (1-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / 0.5**2
            # a = (np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / (np.pi/2)**2
            y = y.squeeze()*a
            return y.unsqueeze(-1)

        return y
    
class ReLUResNet(nn.Module):
    def __init__(self, d, m, degree, test):
        super(ReLUResNet, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.outlayer = nn.Linear(m, 1, bias = False)
        
        self.d = d
        self.degree = degree
        self.test = test
        self.Relu = nn.ReLU()

        Ix = torch.zeros([d,m]).to(device)
        for i in range(d):
            Ix[i,i] = 1
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.Relu(y)**self.degree
        # y = self.fc2(y)
        # y = self.Relu(y)**self.degree
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = self.Relu(y)**self.degree
        # y = self.fc4(y)       
        # y = self.Relu(y)**self.degree
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = self.Relu(y)**self.degree
        # y = self.fc6(y)    
        # y = self.Relu(y)**self.degree
        y = y+s

        y = self.outlayer(y)

        if self.test:
            # a = (1-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / 0.5**2
            a = (np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / (np.pi/2)**2
            y = y.squeeze()*a
            return y.unsqueeze(-1)

        return y


#----------------------------------------------------------------


class TanhResNet2(nn.Module):
    def __init__(self, d, m, test):
        super(TanhResNet2, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.outlayer = nn.Linear(m, 1, bias = False)
        
        self.d = d
        self.test = test
        self.Tanh = nn.Tanh()
        
        Ix = torch.zeros([d,m]).to(device)
        for i in range(d):
            Ix[i,i] = 1
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.Tanh(y)
        # y = self.fc2(y)
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = self.Tanh(y)
        # y = self.fc4(y)       
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = self.Tanh(y)
        # y = self.fc6(y)    
        # y = self.Tanh(y)
        y = y+s
        
        # ones indicate returns identity; adjust factor to tweak effect of neural net's weights/biases
        y = 0 * self.outlayer(y) + torch.ones((x.shape[0],1)).to(device)
        
        if self.test:
            # a = (1-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0]

            # divide for normalization term so peaks at 1
            a = (np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / (np.pi/2)**2

            # this one is closer to 1 everywhere except 0 at boundaries
            # a = ((np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0])**0.1 / (((np.pi / 2)**2)**0.1)

            y = y.squeeze()*a
            return y.unsqueeze(-1)
            # returns [101101, 1] shaped array; need to squeeze to [101101] to 
            # multiply by a, which is shape [101101], then unsqueeze to get back to [101101, 1]. 
            # Then this can be used for grad calculations, but later squeezed back to [101101]
            # and reshaped to [1001, 101].

        return y
    
class TanhResNet3(nn.Module):
    def __init__(self, d, m, test):
        super(TanhResNet3, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.outlayer = nn.Linear(m, 1, bias = False)
        
        self.d = d
        self.test = test
        self.Tanh = nn.Tanh()
        
        Ix = torch.zeros([d,m]).to(device)
        for i in range(d):
            Ix[i,i] = 1
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.Tanh(y)
        # y = self.fc2(y)
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = self.Tanh(y)
        # y = self.fc4(y)       
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = self.Tanh(y)
        # y = self.fc6(y)    
        # y = self.Tanh(y)
        y = y+s
        
        # ones indicate returns identity; adjust factor to tweak effect of neural net's weights/biases
        y = 0 * self.outlayer(y) + torch.ones((x.shape[0],1)).to(device)
        
        if self.test:
            # a = (1-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0]

            # divide for normalization term so peaks at 1
            # a = (np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0] / (np.pi/2)**2

            # this one is closer to 1 everywhere except 0 at boundaries
            # a = ((np.pi-x[:,1])*x[:,1]*(2-x[:,0])*x[:,0])**0.1 / (((np.pi / 2)**2)**0.1)

            # y = y.squeeze()*a
            return y#.unsqueeze(-1)
            # returns [101101, 1] shaped array; need to squeeze to [101101] to 
            # multiply by a, which is shape [101101], then unsqueeze to get back to [101101, 1]. 
            # Then this can be used for grad calculations, but later squeezed back to [101101]
            # and reshaped to [1001, 101].

        return y

class TanhResNet4(nn.Module):
    def __init__(self, d, m, test):
        super(TanhResNet4, self).__init__()
        self.fc1 = nn.Linear(d, m)
        self.fc2 = nn.Linear(m, m)
        self.fc3 = nn.Linear(m, m)
        self.fc4 = nn.Linear(m, m)
        self.fc5 = nn.Linear(m, m)
        self.fc6 = nn.Linear(m, m)
        self.outlayer = nn.Linear(m, 1, bias = False)
        
        self.d = d
        self.test = test
        self.Tanh = nn.Tanh()
        
        Ix = torch.zeros([d,m]).to(device)
        for i in range(d):
            Ix[i,i] = 1
        self.Ix = Ix

    def forward(self, x):
        s = x@self.Ix
        y = self.fc1(x)
        y = self.Tanh(y)
        # y = self.fc2(y)
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc3(y)     
        y = self.Tanh(y)
        # y = self.fc4(y)       
        # y = self.Tanh(y)
        y = y+s
        
        s=y
        y = self.fc5(y)      
        y = self.Tanh(y)
        # y = self.fc6(y)    
        # y = self.Tanh(y)
        y = y+s
        
        y = self.outlayer(y)

        return y