import torch, sys, os, csv, pickle
import torch.nn as nn
import numpy as np
from torch.nn import init
from scipy.io import loadmat
from torch.autograd import Function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def ls_adv_loss(p, v):
    return torch.mean((p-v) ** 2)

def gen_loss(x, x_rec, z_avg, z_log_var, alpha):
    x_rec_loss = torch.mean((x - x_rec).pow(2), dim=-1)
    z_kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_avg.pow(2) - z_log_var.exp(), dim=-1)
    return alpha*z_kl_loss + x_rec_loss

def mean_gen_loss(x, x_rec, z_avg, z_log_var, alpha):
    return torch.mean(gen_loss(x, x_rec, z_avg, z_log_var, alpha))

def n2t(num, tensortype=torch.FloatTensor, gpu=True):
    return tensortype(num).to(device) if gpu else tensortype(num)

def t2n(tensor):
    return tensor.detach().cpu().numpy()

def csvread(file_path):
    return np.loadtxt(file_path,delimiter=',',dtype=np.float32)

def save_model(path, model_name, model):
    check_dir(path)
    torch.save(model.state_dict(), os.path.join(path, '%s.pth'%model_name))

def load_model(path, model_name, model, args=None):
    model.load_state_dict(torch.load(os.path.join(path, '%s.pth'%model_name), map_location='cpu')) # in my computer
    return model

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight, 1.0, 0.02)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias:
                init.constant_(m.bias, 0)

def permute_results(z, x_rec, z_rec, valid_r, mid_r, valid_f, mid_f, z_avg, z_log_var):
    return z.permute(0,2,1), x_rec.permute(0,2,1), z_rec.permute(0,2,1), valid_r.permute(0,2,1), mid_r.permute(0,2,1), valid_f.permute(0,2,1), mid_f.permute(0,2,1), z_avg.permute(0,2,1), z_log_var.permute(0,2,1)

def abnorm_judge(threshold, res):
    if res>threshold:
        return 1
    else:
        return 0

def write_matrix(file_path, matirx):
    with open(file_path, "w+",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matirx)

def txt_save(test_file, filename, num, losses, type='batch'):
    file = open(filename,'a')
    s = '%s %s %d |'%(test_file, type, num)
    for key, loss in zip(losses.keys(), losses.values()):
        s += ' %s = %.4f |'%(key, loss)  
    s += '\n'
    file.write(s)

def acc_save(filename, e, acc):
    data_file = open(filename, 'a+')#, newline='')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['e%d'%e, '%.4f'%acc])
    data_file.close()

def test_save(filename, file_name, loss, time, flag):
    data_file = open(filename, 'a+')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow([file_name[:-4], '%.4f'%t2n(loss), '%.4f'%(time*0.1), flag])
    data_file.close()

def loss_save(filename, rec_loss):
    data_file = open(filename, 'a+')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['%.4f'%rec_loss])
    data_file.close()

def get_files(test=False):
    label=[0,1,2,3,4,5,6,7,8]   
    path = './data/UoC/DataForClassification_TimeDomain.mat'
    data = loadmat(path)
    data = data['AccTimeDomain']
    da = []
    lab = []
    
    start,end=0,104  
    i=0
    while end <= data.shape[1]:
        data1 = data[:,start:end]
        data1 = data1.reshape(-1,1)
        da1 = data1
        lab1 = np.ones_like(da1)*label[i]
        da.append(da1)
        lab.append(lab1)
        start += 104
        end += 104
        i += 1
    return [da,lab]
