import os,argparse,torch,sys,csv,datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data.dataset import DB
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.tcn_gan import Generator, Discriminator
from models.tools import csvread, check_dir, save_model, load_model, ls_adv_loss, mean_gen_loss, permute_results, txt_save, acc_save, test_save, loss_save
import warnings
warnings.filterwarnings("ignore") 
starttime = datetime.datetime.now()

parser = argparse.ArgumentParser(description='Arguments parser for TCN-GAN training')
parser.add_argument('-dx','--dim_x', type=int, default=2)
parser.add_argument('-dz','--dim_z', type=int, default=8)
parser.add_argument('-sl','--seqlen', type=int, default=120)
parser.add_argument('-c','--cons', type=int, nargs='+', default=[32,64])
parser.add_argument('-cdis','--cons_dis', type=int, nargs='+', default=[8,16])
parser.add_argument('-db','--dataset_type', type=str, default='bear')
parser.add_argument('-cm','--code_mode', type=str, default='train_mode')

parser.add_argument('-al','--alpha', type=float, default=0.1) 
parser.add_argument('-wr','--w_rec', type=float, default=1.0)
parser.add_argument('-we','--w_enc', type=float, default=0.001)
parser.add_argument('-wa','--w_adv', type=float, default=1.0)
parser.add_argument('-wg','--w_gan', type=float, default=0.1)
parser.add_argument('-wd','--w_dis', type=float, default=10)

parser.add_argument('-t','--trails', type=int, default=3)
parser.add_argument('-b','--batch_size', type=int, default=16)
parser.add_argument('-e','--epochs', type=int, default=15)
parser.add_argument('-o','--output', type=str, default='tmp')
parser.add_argument('-dev','--device', type=int, default=0)
parser.add_argument('-use','--data_use', type=float, default=0.01)
parser.add_argument('-train','--train_file', type=int, default=97)
parser.add_argument('-test','--test_dir', type=int, default=0)
parser.add_argument('-threshold','--threshold', type=float, default=1.9)
args = parser.parse_args()
print('>> %s\n' % str(args))

lr = 5.0e-5
betas = (0.5, 0.999)
check_dir('./output/%s'%args.output)
mse = torch.nn.MSELoss(reduce=True, size_average=True)
device = torch.device("cuda:%d"%args.device if torch.cuda.is_available() else "cpu")
print(device)

def get_losses(generator, discriminator, x):
    real_label = Variable(torch.FloatTensor(x.shape[0], args.seqlen, 1).fill_(1.0), requires_grad=False).to(device)
    fake_label = Variable(torch.FloatTensor(x.shape[0], args.seqlen, 1).fill_(0.0), requires_grad=False).to(device)
    losses = {}
    
    z, x_rec, z_rec, z_avg, z_log_var = generator(x)
    valid_r, mid_r = discriminator(x.permute(0,2,1))
    valid_f, mid_f = discriminator(x_rec)
    z, x_rec, z_rec, valid_r, mid_r, valid_f, mid_f, z_avg, z_log_var = permute_results(z, x_rec, z_rec, valid_r, mid_r, valid_f, mid_f, z_avg, z_log_var)
    
    rec_loss = mean_gen_loss(x, x_rec, z_avg, z_log_var, args.alpha)
    adv_loss = mse(mid_r, mid_f)
    enc_loss = mse(z, z_rec)
    gan_loss = ls_adv_loss(valid_f, real_label)
    dis_loss = ls_adv_loss(valid_r, real_label) + ls_adv_loss(valid_f, fake_label) 
    loss = args.w_rec*rec_loss + args.w_adv*adv_loss + args.w_gan*gan_loss + args.w_enc*enc_loss

    losses['loss'] = loss
    losses['rec_loss'] = args.w_rec*rec_loss
    losses['adv_loss'] = args.w_adv*adv_loss
    losses['enc_loss'] = args.w_enc*enc_loss
    losses['gan_loss'] = args.w_gan*gan_loss
    losses['dis_loss'] = args.w_dis*dis_loss
    return losses, x, x_rec, z, z_rec

def test_model(file, filename, test_loader, generator, discriminator, threshold):
    cnt = 0
    for i,batch_data in enumerate(test_loader):
        x, y = batch_data
        losses, x, x_rec, z, z_rec = get_losses(generator, discriminator, x)
        txt_save(file, filename, i, losses)
        if losses['rec_loss'] < threshold:
            cnt = 0
        if losses['rec_loss'] >= threshold:
            cnt += 1
        if cnt==5:
            return losses, i+1, 1
    return losses, 0, 0

train_data_path = './data/CWRU/Normal_Baseline_Data/%d.csv'%args.train_file
data = csvread(train_data_path)
train_data_len = int(len(data)*args.data_use)
train_data = torch.from_numpy(data)[:train_data_len,:].to(device)
data_len = train_data.shape[0]
print('data_len: %d'%train_data.shape[0])
train_db = DB(train_data, args.seqlen)
train_loader = DataLoader(dataset=train_db, batch_size=args.batch_size, shuffle=True, drop_last=True)

for e in range(args.trails):
    generator = Generator(args.dim_x, args.dim_z, args.cons).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    discriminator = Discriminator(args.dim_x, args.cons_dis).to(device)
    optimizer_dis = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    print(generator)
    print(discriminator)

    if args.code_mode == 'train_mode':
        print("\ntrain")
        for epochs in range(args.epochs):
            for i,batch_data in enumerate(train_loader):
                x, y = batch_data
                optimizer_dis.zero_grad()
                losses, x, x_rec, z, z_rec = get_losses(generator, discriminator, x)
                losses['dis_loss'].backward()
                optimizer_dis.step()
                optimizer.zero_grad()
                losses, x, x_rec, z, z_rec = get_losses(generator, discriminator, x)
                losses['loss'].backward()
                optimizer.step()
            txt_save('train_file', './output/%s/train_e%d.txt'%(args.output, e), epochs, losses, 'epoch')
            loss_save('./output/%s/rec_loss_e%d.csv'%(args.output, e), losses['rec_loss'])
        save_model('./output/%s/'%args.output, 'tcn_gan_generator_%s_wonorm_e%d'%(args.dataset_type, e), generator)
        save_model('./output/%s/'%args.output, 'tcn_gan_discriminator_%s_wonorm_e%d'%(args.dataset_type, e), discriminator)

    if args.code_mode=='test_mode':  
        generator = load_model('./output/%s/'%args.output, 'tcn_gan_generator_%s_wonorm_e%d'%(args.dataset_type,e), generator)
        discriminator = load_model('./output/%s/'%args.output, 'tcn_gan_discriminator_%s_wonorm_e%d'%(args.dataset_type,e), discriminator)
        threshold = args.threshold * torch.from_numpy(csvread('./output/%s/rec_loss_e%d.csv'%(args.output, e)))[-1].to(device)  

        normal_test_data = torch.from_numpy(data)[train_data_len:2*train_data_len,:].to(device)
        test_db = DB(normal_test_data, args.seqlen)
        test_loader = DataLoader(dataset=test_db, batch_size=args.batch_size, shuffle=False, drop_last=True)
        losses, time, flag = test_model('%d'%args.train_file, './output/%s/test_%s_e%d.txt'%(args.output, 'normal', e), test_loader, generator, discriminator, threshold)
        test_save('./output/%s/test_e%d.csv'%(args.output, e), '%d.csv'%args.train_file, losses['rec_loss'], time, flag)

        files = os.listdir('./data/CWRU/%s/'%args.test_dir)
        abnorm_cnt = 0.0
        abnorm_sum = len(files)
        for file in files:
            test_data = csvread('./data/CWRU/%s/%s'%(args.test_dir, file))
            test_data = torch.from_numpy(test_data)[:data_len].to(device)
            test_db = DB(test_data, args.seqlen)
            test_loader = DataLoader(dataset=test_db, batch_size=args.batch_size, shuffle=False, drop_last=True)
            losses, time, flag = test_model(file, './output/%s/test_%s_e%d.txt'%(args.output, 'abnormal', e), test_loader, generator, discriminator, threshold)
            test_save('./output/%s/test_e%d.csv'%(args.output, e), file, losses['rec_loss'], time, flag)
            abnorm_cnt += flag
        acc = abnorm_cnt/abnorm_sum
        print('\nabnormal accuracy: %.4f'%acc)
        acc_save('./output/%s/acc.csv'%args.output, e, acc)
    endtime = datetime.datetime.now()
    print (endtime-starttime)
