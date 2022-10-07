import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import classes
import torch.nn.functional as F
import torch.nn.init
import torchvision
import csv
import math
from corr import compute_corr
from scipy import stats
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch_poly_lr_decay import PolynomialLRDecay


torch.manual_seed(42)
torch.cuda.manual_seed(42)

def return_batch_read(batches,datadir):
    final = []
    for i in range(len(batches)):
        data = np.load(datadir+batches[i])
        data = data.astype(np.float32)
        final.append(data)
    return final

def save_attention_weights(filenames,attention_weights):
    attn_path = './attention_weights/' 
    for i in range(len(filenames)):
        np.save(attn_path+filenames[i][:-4]+'-attn.npy',attention_weights[i])
    return



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(sys.argv)!=3:
        print("python predictor.py <data-directory> <epochs>")
        sys.exit()

    N_FOLDS = 10

    BATCH = 128 # 128
    DROPOUT = 0.50 
    LR = 0.01 #0.00005  #0.0001       
    LR_INIT = 0.001  
    EPOCHS = int(sys.argv[2])

    datadir = sys.argv[1]
    fold_path = './data/crossval-folds/'

    FOLDS_LOADER_TRAIN = []
    FOLDS_LOADER_TEST = []

    for i in range(N_FOLDS):
        #fold_file_train = fold_path + 'TRAIN_FOLD_' + str(i+1) + '_FULL_FEAT.csv' #'_TEN_MT.csv'
        #fold_file_test = fold_path + 'TEST_FOLD_' + str(i+1) + '_FULL_FEAT.csv' #'_TEN_MT.csv'
        fold_file_train = fold_path + 'DAP_MEANSTD_TRAIN_' + str(i+1) + '.csv'
        fold_file_test =  fold_path + 'DAP_MEANSTD_TEST_' + str(i+1) + '.csv'
        
        sequential = classes.load_fold
        dataset_train = sequential(fold_file_train,datadir)
        FOLDS_LOADER_TRAIN.append(DataLoader(dataset=dataset_train, batch_size=BATCH,shuffle=True,drop_last=True))
        dataset_test = sequential(fold_file_test,datadir)
        FOLDS_LOADER_TEST.append(DataLoader(dataset=dataset_test, batch_size=1))

    TEST_OUTPUT = []
    TEST_OUTPUT_VAR = []
    TEST_FILES = []
    TEST_REF = []


    for fold in range(N_FOLDS):
            
        print('FOLD NR: {}'.format(fold+1))

        model = classes.TRANSFORMER(n_gru_layers=3,hidden_dim=100,batch_size=BATCH).cuda()
        criterion = nn.MSELoss()

        log_var_a = torch.ones((1,), requires_grad=True)
        log_var_b = torch.ones((1,), requires_grad=True)
        params = ([p1 for p1 in model.parameters()] + [log_var_a] + [log_var_b])
        optimizer = torch.optim.Adam(params, lr=LR) #, weight_decay = 0.01)


        scheduler = PolynomialLRDecay(optimizer, max_decay_steps=50, end_learning_rate=0.001, power=0.9)
        

        early_stopping = classes.EarlyStopping()
        
        epoch_counter = int(sys.argv[2])    
        for ep in range(epoch_counter):
            model.train()
            train_loss = []
            test_loss = []


            for batch_idx,  (file_train, y1) in enumerate(FOLDS_LOADER_TRAIN[fold]):
              
                x = return_batch_read(file_train,datadir)
                   
                optimizer.zero_grad()  

                len_x = []
                for i in range(0,len(x)):
                    x[i] = torch.Tensor(x[i])
                    len_x.append(len(x[i]))
                        
                x = pad_sequence(x, batch_first=True)
                # [Batch, Len, Feats]
                    
                x = pack_padded_sequence(x,len_x,batch_first=True, enforce_sorted=False)
                x = x.cuda()

                output1, attn_weights  = model(x, len_x)

                loss = criterion(output1.squeeze(),y1.float().cuda().squeeze())
                
                    
                train_loss.append(loss.cpu().detach().numpy())   
                loss.backward()    
                optimizer.step()
                
                
            early_stopping(sum(train_loss))
            model.eval()
            for batch_idx, (file_test, y1_test) in enumerate(FOLDS_LOADER_TEST[fold]):
                x_test = return_batch_read(file_test,datadir)
                len_x_test = []
                for j in range(0,len(x_test)):
                    x_test[j] = torch.Tensor(x_test[j])
                    len_x_test.append(len(x_test[j]))

                x_test = pad_sequence(x_test, batch_first=True)
                x_test = pack_padded_sequence(x_test,len_x_test,batch_first=True, enforce_sorted=False)
                x_test = x_test.cuda()
                    
                y1_test = y1_test.cuda()
                y2_test = y2_test.cuda()
                y3_test = y3_test.cuda()

                output_test1, attn_weights  = model(x_test, len_x_test)
              
                # Saves the attention weights only at the last epoch
                if (ep == int(sys.argv[2])-1):
                    save_attention_weights(file_test,attn_weights.cpu().detach().numpy())  
                    
                loss_test1 = criterion(output_test1.squeeze(),y1_test.float().squeeze())
                loss_test2 = criterion(output_test2.squeeze(),y2_test.float().squeeze())
                loss_test3 = criterion(output_test3.squeeze(),y3_test.float().squeeze())

                loss = loss_test1

                test_loss.append(loss.cpu().detach().numpy())

                if( ep == epoch_counter-1 or early_stopping.early_stop):
                    TEST_FILES.append(str(file_test[0]))              # first index of the tuple gives the filename
                    TEST_REF.append(round(float(y1_test.cpu().numpy().squeeze()),3))
                    TEST_OUTPUT.append(round(float(output_test1.detach().cpu().numpy().squeeze()),3))
                    TEST_OUTPUT_VAR.append(round(float(output_test2.detach().cpu().numpy().squeeze()),3))   

            scheduler.step()

            print('Epoch: {} -- Train Loss: {} -- Validation Loss: {}'.format(ep,
                   np.mean(train_loss),np.mean(test_loss)))

            if (early_stopping.early_stop):
                break


    RESULTS = list(zip(TEST_FILES,TEST_REF,TEST_OUTPUT,TEST_OUTPUT_VAR))

    with open('output_int_pred.csv', 'w') as fp:
        fp.write('\n'.join('%s,%s,%s,%s' % x for x in RESULTS))
    fp.close()

    compute_corr('output_int_pred.csv',52)
    
