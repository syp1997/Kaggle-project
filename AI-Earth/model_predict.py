import os
import torch
from copy import deepcopy
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn import preprocessing
import zipfile
import shutil
from sklearn.preprocessing import StandardScaler
import math

def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    
class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):   
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), self.data['label'][idx]
    
    
def fit_data(data_list, fit=True):
    a,b,c,d = data_list[0].shape
    all_data = []
    for data in data_list:
        new_data = data.reshape(-1)
        all_data.append(new_data)
    all_data = np.stack(all_data,1)
    print(all_data.shape)
    if fit:
        standardScaler.fit(all_data)
        print("fit train data")
    all_data = standardScaler.transform(all_data)
    res_data = []
    for i in range(all_data.shape[1]):
        data = all_data[:,i].reshape(a,b,c,d)
        res_data.append(data)
    return res_data

def load_data():
    # CMIP data    
    train = xr.open_dataset('/tcdata/enso_round1_train_20210201/CMIP_train.nc')
    label = xr.open_dataset('/tcdata/enso_round1_train_20210201/CMIP_label.nc')    
   
    train_sst = train['sst'][:, :12].values  # (4645, 12, 24, 72)
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    train_label = label['nino'][:, 12:36].values

    train_ua = np.nan_to_num(train_ua) # trans nan to 0
    train_va = np.nan_to_num(train_va)
    train_t300 = np.nan_to_num(train_t300)
    train_sst = np.nan_to_num(train_sst)
    
#     data_list = [train_sst,train_t300,train_ua,train_va]
#     train_sst,train_t300,train_ua,train_va = fit_data(data_list, fit=True)

    # SODA data    
    train2 = xr.open_dataset('/tcdata/enso_round1_train_20210201/SODA_train.nc')
    label2 = xr.open_dataset('/tcdata/enso_round1_train_20210201/SODA_label.nc')
    
    train_sst2 = train2['sst'][:, :12].values  # (100, 12, 24, 72)
    train_t3002 = train2['t300'][:, :12].values
    train_ua2 = train2['ua'][:, :12].values
    train_va2 = train2['va'][:, :12].values
    train_label2 = label2['nino'][:, 12:36].values
    
    train_sst2 = np.nan_to_num(train_sst2) # trans nan to 0
    train_t3002 = np.nan_to_num(train_t3002)
    train_ua2 = np.nan_to_num(train_ua2)
    train_va2 = np.nan_to_num(train_va2)
    
#     data_list = [train_sst2,train_t3002,train_ua2,train_va2]
#     train_sst2,train_t3002,train_ua2,train_va2 = fit_data(data_list, fit=False)

    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(train_label2)))

    dict_train = {
        'sst':train_sst,
        't300':train_t300,
        'ua':train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst':train_sst2,
        't300':train_t3002,
        'ua':train_ua2,
        'va': train_va2,
        'label': train_label2}
    train_dataset = EarthDataSet(dict_train)
    valid_dataset = EarthDataSet(dict_valid)
    return train_dataset, valid_dataset

    
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        resnet = models.resnet18()
        resnet.conv1 = nn.Conv2d(12, 64, kernel_size=(4, 8), stride=(1, 1), padding=(2, 4), bias=False)
        resnet.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnet1 = nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet2 = nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet3 = nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet4 = nn.Sequential(*(list(resnet.children())[:-1]))
        
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(4,8))) 
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(4,8))) 
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(4,8))) 
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(4,8))) 
        
        self.batch_norm = nn.BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True)
#         self.lstm = nn.LSTM(input_size = 1540 * 4, hidden_size = 256, num_layers=2, batch_first=True, bidirectional=False)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,64))
        self.linear0 = nn.Linear(2048, 64)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(64, 24)

    def forward(self, sst, t300, ua, va):
        sst = self.resnet1(sst).squeeze(-1).squeeze(-1)  # batch * 12 * (24 - 2) * (72 -2)
        t300 = self.resnet2(t300).squeeze(-1).squeeze(-1)
        ua = self.resnet3(ua).squeeze(-1).squeeze(-1)
        va = self.resnet4(va).squeeze(-1).squeeze(-1)
        
#         sst = torch.flatten(sst, start_dim=1)  # batch * 12 * 1540
#         t300 = torch.flatten(t300, start_dim=1)
#         ua = torch.flatten(ua, start_dim=1)
#         va = torch.flatten(va, start_dim=1)  
        
        x = torch.cat([sst, t300, ua, va], dim=-1) # batch * 12 * (1540 * 4)
#         x = torch.flatten(x, start_dim=1)
#         x, (h_n, c_n) = self.lstm(x)
#         x = x[:,-1]
#         x = self.avgpool(x.unsqueeze(-2)).squeeze(dim=-2)
        x = self.linear0(x)
        x = self.tanh(x)
        x = self.linear(x)
        return x
    
# make zip
def make_zip(res_dir='./result', output_dir='result.zip'):  
    z = zipfile.ZipFile(output_dir, 'w')  
    for file in os.listdir(res_dir):  
        if '.npy' not in file:
            continue
        z.write(res_dir + os.sep + file)  
    z.close()
    
    
if __name__ == '__main__': 
    print("Start run")
    set_seed()
#     standardScaler = StandardScaler()
#     train_dataset, valid_dataset = load_data()      
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    model = CNN_Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    loss_fn = nn.MSELoss()   

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    model.load_state_dict(torch.load('models/basemodel_best.pt'))
    
    test_path = '/tcdata/enso_round1_test_20210201/'

    ### load test data
    files = os.listdir(test_path)
    test_feas_dict = {}
    for file in files:
        test_feas_dict[file] = np.load(test_path + file)
        
    ### 2. predict
    test_predicts_dict = {}
    for file_name, val in test_feas_dict.items():
        SST = np.expand_dims(val[:,:,:,0],axis=0)
        T300 = np.expand_dims(val[:,:,:,1],axis=0)
        Ua = np.expand_dims(val[:,:,:,2],axis=0)
        Va = np.expand_dims(val[:,:,:,3],axis=0)
        
        SST = np.nan_to_num(SST) # trans nan to 0
        T300 = np.nan_to_num(T300)
        Ua = np.nan_to_num(Ua)
        Va = np.nan_to_num(Va)
        
#         data_list = [SST,T300,Ua,Va]
#         SST,T300,Ua,Va = fit_data(data_list, fit=False)

        SST = torch.tensor(SST).to(device).float()
        T300 = torch.tensor(T300).to(device).float()
        Ua = torch.tensor(Ua).to(device).float()
        Va = torch.tensor(Va).to(device).float()
        
        result = model(SST, T300, Ua, Va).view(-1).detach().cpu().numpy()
        test_predicts_dict[file_name] = result
        
    ### 3. save results
    if os.path.exists('./result/'):  
        shutil.rmtree('./result/', ignore_errors=True)  
    os.makedirs('./result/')
    for file_name, val in test_predicts_dict.items(): 
        np.save('./result/' + file_name, val)
        
    make_zip()
    print("End")