import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
import time
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import math

import preparing_model
from preparing_model import ResizeBinarize
from preparing_model import add_extended_class_column
from preparing_model import ThreePages

def main():
    PAGE_IMGS_PATH = './base/SinglePageTIF/'
    TRAIN_LABEL_PATH = './base/train.csv'
    TEST_LABEL_PATH = './base/test.csv'

    df_train = pd.read_csv(TRAIN_LABEL_PATH, sep=';', skiprows=0, low_memory=False)
    df_test = pd.read_csv(TEST_LABEL_PATH,sep=';', skiprows=0, low_memory=False)

    transform = transforms.Compose([
                            ResizeBinarize(),
                            #transforms.Resize(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
                            #transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
                    ])
    
    df_train = add_extended_class_column(df_train)
    df_test = add_extended_class_column(df_test)

    df_val = df_train.iloc[-200:,:]
    df_train = df_train.iloc[:-200,:]

    label2Idx = {'single page': 3,'first of many': 2,'middle': 1,'last page':0}

    train_data = ThreePages(df_train,PAGE_IMGS_PATH, label2Idx, transform )
    valid_data = ThreePages(df_val,PAGE_IMGS_PATH, label2Idx, transform )
    test_data = ThreePages(df_test,PAGE_IMGS_PATH, label2Idx, transform )

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of test examples: {len(test_data)}')

    train_iterator = data.DataLoader(train_data, 
                                    batch_size = preparing_model.BATCH_SIZE, shuffle=False)

    valid_iterator = data.DataLoader(valid_data, 
                                    batch_size = preparing_model.BATCH_SIZE)
    test_iterator = data.DataLoader(test_data, 
                                    batch_size = preparing_model.BATCH_SIZE)

    create_model(train_iterator, valid_iterator, test_iterator)

def create_model(train_iterator, valid_iterator, test_iterator):
    # Train model
    FOUND_LR = 5e-3
    model = preparing_model.MODEL
    
    params = [
            {'params': model.base_model.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.classifier.parameters()}
            ]

    optimizer = optim.Adam(params, lr = FOUND_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    #Train    
    EPOCHS = 2
    best_valid_loss = float('inf')
    experiment_data = []
    for epoch in range(EPOCHS):
        
        start_time = time.monotonic()
        
        train_loss, train_acc, train_kappa, train_acc_2, train_kappa_2 = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, valid_kappa, valid_acc_2, valid_kappa_2 = evaluate(model, valid_iterator, criterion, device)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print (f'Winner Epoch: {epoch+1:02}.')
            torch.save(model.state_dict(), './model/'+ preparing_model.EXPERIMENT + '.pt')
            
        scheduler.step(valid_loss)

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train Kappa: {train_kappa*100:.2f}% | Train Acc 2: {train_acc_2*100:.2f}% | Train Kappa 2: {train_kappa_2*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Val. Kappa: {valid_kappa*100:.2f}% |  Val. Acc 2: {valid_acc_2*100:.2f}% |  Val. Kappa 2: {valid_kappa_2*100:.2f}%')

        experiment_data.append([train_loss, valid_loss, train_acc, valid_acc, train_kappa, valid_kappa, train_acc_2, valid_acc_2, train_kappa_2, valid_kappa_2, start_time, end_time])
        experiment_df = pd.DataFrame(experiment_data,columns=['train_loss','valid_loss','train_acc', 'valid_acc','train_kappa', 'valid_kappa', 'train_acc_2', 'valid_acc_2', 'train_kappa_2', 'valid_kappa_2', 'start', 'end' ])
        experiment_df.to_pickle('./model/'+ preparing_model.EXPERIMENT+ '.pk')
    
    experiment_df = pd.read_pickle('./model/'+ preparing_model.EXPERIMENT+ '.pk')
    print(preparing_model.EXPERIMENT)
    print(experiment_df.sort_values(by=['valid_loss']).iloc[0])
    train_loss, valid_loss, train_acc,valid_acc,test_kappa,valid_kappa,train_acc_2,valid_acc_2,train_kappa_2,valid_kappa_2,start,end = experiment_df.sort_values(by=['valid_loss']).iloc[0]

    model.load_state_dict(torch.load('./model/'+ preparing_model.EXPERIMENT + '.pt'))
    test_loss, test_acc, test_kappa, test_acc_2, test_kappa_2 = evaluate(model, test_iterator, criterion, device)
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}% |  Val. Kappa: {test_kappa*100:.2f}% |  Val. Acc 2: {test_acc_2*100:.2f}% |  Val. Kappa 2: {test_kappa_2*100:.2f}%')

def calculate_accuracy(y_pred, y): 
    correct = y_pred.eq(y.view_as(y_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

#train loop
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_2 = 0
    epoch_kappa = 0
    epoch_kappa_2 = 0
    
    model.train()
    
    for (x, y, name) in iterator:
        
        x1, x2, x3 = x
        
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        
        y = y.to(device)
        
        optimizer.zero_grad()
                
        y_pred = model(x1, x2, x3)
        
        loss = criterion(y_pred, y)
        
        top_pred = y_pred.argmax(1, keepdim = True)
        
        acc = calculate_accuracy(top_pred, y)
        acc_2 = calculate_accuracy(torch.div(top_pred, 2, rounding_mode='trunc'), torch.div(y, 2, rounding_mode='trunc'))

        kappa = cohen_kappa_score(y.tolist(), top_pred.flatten().cpu())
        kappa_2 = cohen_kappa_score(torch.div(y, 2, rounding_mode='trunc').tolist(), torch.div(top_pred.flatten().cpu(), 2, rounding_mode='trunc'))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_acc_2 += acc_2.item()
        epoch_kappa += 1.0 if math.isnan(kappa) else kappa
        epoch_kappa_2 += 1.0 if math.isnan(kappa_2) else kappa_2
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_kappa / len(iterator), epoch_acc_2 / len(iterator), epoch_kappa_2 

#Train evaluate
def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_2 = 0
    epoch_kappa = 0
    epoch_kappa_2 = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y, _) in iterator:
            
            x1, x2, x3 = x
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)            
            y = y.to(device)

            y_pred = model(x1, x2, x3)

            loss = criterion(y_pred, y)

            top_pred = y_pred.argmax(1, keepdim = True)

            acc = calculate_accuracy(top_pred, y)
            acc_2 = calculate_accuracy(torch.div(top_pred, 2, rounding_mode='trunc'), torch.div(y, 2, rounding_mode='trunc'))

            kappa = cohen_kappa_score(y.tolist(), top_pred.flatten().cpu())
            kappa_2 = cohen_kappa_score(torch.div(y, 2, rounding_mode='trunc').tolist(), torch.div(top_pred.flatten().cpu(), 2, rounding_mode='trunc'))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_acc_2 += acc_2.item()
            epoch_kappa += 1.0 if math.isnan(kappa) else kappa
            epoch_kappa_2 += 1.0 if math.isnan(kappa_2) else kappa_2
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_kappa / len(iterator), epoch_acc_2 / len(iterator), epoch_kappa_2 / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    main()
    