

import CoAtNet_36
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import os
import re
import glob
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

SAMPLE_RATE         = 44100
N_MEL               = 64
N_FFT               = 1024
N_HOP               = 192
TRAIN_DATA_PATH     = 'dataset/train'
VAL_DATA_PATH       = 'dataset/val'
TEST_DATA_PATH      = 'dataset/test'
MODEL_PATH          = 'models/model_4.pt'
ATTACK_DATA_PATH    = 'attack_files/helloworld0318.wav'
AUG_MASK            = 1
AUG_NOISE_GAIN      = 0
BATCH_SIZE          = 64
N_EPOCH             = 1000
LR                  = 0.001
ATTACK_OUT_RANK     = 4




def key_to_label(key):

    keys = "0123456789abcdefghijklmnopqrstuvwxyz"
    label = keys.index(key)
    
    return label

def detect_keypoint(wv, th=0.1, lock=8000):
    wv = wv[0]
    kp = []
    key_marker = []
    i = 0
    while i < len(wv):
        if np.abs(wv[i]) > th:
            kp.append(i)
            for j in range(lock):
                key_marker.append(th)
            i += lock 
        else:
            key_marker.append(0)
            i += 1
    return kp,key_marker

def get_melspectrogram(wv):
    # 引数：1*16384のtensor
    # 出力：64*64のスペクトログラム
    spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MEL,
        win_length=None,
        hop_length=N_HOP,
        window_fn=torch.hann_window,
        power=2.0,)
    win = torch.hann_window(16384)
    spec = spectrogram(wv).numpy()[0]
    spec = np.delete(spec,64,1)
    spec = np.log(spec)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
    spec = spec[:,11:11+64] #中央部分をカット
    return spec

def make_dataset(path, num_mask=1, noise_gain=0):
    
    files = glob.glob(path+'/*')
    num = len(files)
    
    X = torch.zeros(num*num_mask,1,64,64, dtype=torch.float32)
    y = torch.zeros(num*num_mask, dtype=torch.int64)
    
    
    for filecount,file in enumerate(tqdm(files)):
        #print(file)
        waveform, sample_rate = torchaudio.load(uri=file)
        
        spec = get_melspectrogram(waveform)
        
        #plt.plot(waveform)
        #plt.imshow(spec)
        #plt.show()
        
        label = re.search(r'\\(.+).',file).group(1)    # \ と \ に挟まれた部分を検索し，int型に変換
        label = key_to_label(str(label[0]))
        
        X[filecount] = torch.from_numpy(spec)
        y[filecount] = label
        
    dataset = torch.utils.data.TensorDataset(X,y)
    
    return dataset

def make_dataset_all(path, hop):
    
    spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MEL,
        win_length=None,
        hop_length=N_HOP,
        window_fn=torch.hann_window,
        power=2.0,
    )
    
    waveform, sample_rate = torchaudio.load(uri=path)
    num_phrases = int((waveform.shape[1]-16384)/hop)
    
    print(waveform.shape)
    print(num_phrases)
    plt.plot(waveform[0])
    plt.show()
    
    
    X = torch.zeros(num_phrases,1,64,64, dtype=torch.float32)
    y = torch.zeros(num_phrases, dtype=torch.int64)
    
    win = torch.hann_window(16384)
    
    i = 0
    for i in tqdm(range(num_phrases)):
        spec = spectrogram(win*waveform[:,hop*i:hop*i+16384]).numpy()[0]
        spec = np.delete(spec,64,1)
        spec = np.log(spec)
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
        
        X[i] = torch.from_numpy(spec[:,11:11+64])
        y[i] = 9999 # 適当な数字
        
    dataset = torch.utils.data.TensorDataset(X,y)
    
    return dataset,waveform

def make_dataset_points(path):

    waveform, sample_rate = torchaudio.load(uri=path)
    plt.plot(waveform[0])
    plt.grid()
    plt.show()
    th_input = input('threshod(float) : ')
    
    key_point, key_marker = detect_keypoint(waveform, th=float(th_input))
    frame_num = len(key_point)
    print(f"{frame_num} keys ware detected.")
    
    X = torch.zeros(frame_num,1,64,64, dtype=torch.float32)
    y = torch.zeros(frame_num, dtype=torch.int64)

    for i,p in enumerate(tqdm(key_point)):
        # キー検出点をもとにフレーズで抽出，5000をピーク
        phrase = waveform[:,p-5000:p+11384]
        spec = get_melspectrogram(phrase)
        X[i] = torch.from_numpy(spec)
        #print(f'Dataset creating... {i+1}/{frame_num}')

    dataset = torch.utils.data.TensorDataset(X,y)

    return dataset, key_marker

def learn():
    
    train_set = make_dataset(TRAIN_DATA_PATH, AUG_MASK, AUG_NOISE_GAIN)
    val_set   = make_dataset(VAL_DATA_PATH,   AUG_MASK, AUG_NOISE_GAIN)
    print("Datasets were created.")
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader  = torch.utils.data.DataLoader(val_set,  batch_size=BATCH_SIZE, shuffle=True)

    # 学習記録の保存
    history = {"train_loss":[],"validation_loss":[],"train_acc":[],"validation_acc":[]}

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1,2,figsize=(9,4))
    ax_loss,ax_acc = ax[0], ax[1]
    ax_loss.plot(history["train_loss"])
    ax_loss.plot(history["validation_loss"])
    ax_acc.plot(history["train_acc"])
    ax_acc.plot(history["validation_acc"])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')

    net = CoAtNet_36.coatnet_0().to(device)

    print(net)

    #最適化方法の設定
    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)

    loss_func = nn.CrossEntropyLoss()
  
    # 学習
    best_accuracy = 0.0
    print("Start...")
    for e in range(N_EPOCH):
        loss = None
        train_loss = 0.0
        accuracy_train = 0.0
        net.train()
        for i,(data,labels) in enumerate(tqdm(train_loader)):
            data,labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = loss_func(output,labels)
            train_loss += loss.item()
            predict = output.argmax(dim=1,keepdim=True)
            accuracy_train += predict.eq(labels.view_as(predict)).sum().item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        accuracy_train /= len(train_loader.dataset)
        print("Epoch : {}".format(e))
        print(f"LR : {scheduler.get_lr()}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(accuracy_train)
        scheduler.step()
        
        # 検証
        net.eval()
        val_loss = 0.0
        accuracy_val = 0.0
        with torch.no_grad():
            for data,labels in val_loader:
                data,labels = data.to(device),labels.to(device)
                #順伝搬の計算
                output = net(data)
                loss = loss_func(output,labels)
                val_loss += loss.item()
                predict = output.argmax(dim=1,keepdim=True)
                accuracy_val += predict.eq(labels.view_as(predict)).sum().item()
        val_loss /= len(val_loader.dataset)
        accuracy_val /= len(val_loader.dataset)
        if accuracy_val > best_accuracy:
            torch.save(net.state_dict(), MODEL_PATH)
            best_accuracy = accuracy_val
        
        print("Train loss: {}, Validation loss: {}, Train accuracy: {}, Validation accuracy: {}, Best: {}\n"
              .format(train_loss,val_loss,accuracy_train,accuracy_val,best_accuracy))
        
        history["validation_loss"].append(val_loss)
        history["validation_acc"].append(accuracy_val)
        
        # 進捗描画
        ax_loss.cla()
        ax_loss.plot(history["train_loss"], label="train loss")
        ax_loss.plot(history["validation_loss"], label="validation loss")
        ax_loss.set_ylabel("loss")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylim(0,)
        ax_loss.legend()
        
        ax_acc.cla()
        ax_acc.plot(history["train_acc"], label="train accuracy")
        ax_acc.plot(history["validation_acc"], label="validation accuracy")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylim(0,1)
        ax_acc.legend()
        plt.pause(0.1)
        
    
    #結果
    #print(history)
    plt.figure()
    plt.plot(range(1, N_EPOCH+1), history["train_loss"], label="train_loss")
    plt.plot(range(1, N_EPOCH+1), history["validation_loss"], label="validation_loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("img/loss.png")

    plt.figure()
    plt.plot(range(1, N_EPOCH+1), history["validation_acc"])
    plt.title("test accuracy")
    plt.xlabel("epoch")
    plt.savefig("img/test_acc.png")

def test():
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')

    net = CoAtNet_36.coatnet_0().to(device)
    
    attack_set = make_dataset(TEST_DATA_PATH, AUG_MASK)
    attack_loader = torch.utils.data.DataLoader(attack_set, batch_size=BATCH_SIZE, shuffle=False)
    
    params = torch.load(MODEL_PATH,map_location=torch.device(device))
    
    net.load_state_dict(params)

    net.eval()
    
    acc = 0
    n = 0
    for i, (data,labels) in enumerate(attack_loader):
        data = data.to(device)
        labels = labels.to(device)
        output = net(data)
        output_label = torch.max(output,1)[1] #予測ラベル
        acc += (output_label == labels).sum().item()
        n += len(labels)
        print("predict: ",output_label)
        print("label  : ",labels)

    print("Acc: ",(acc/n))
    
    
    return

def attack():
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')
    
    net = CoAtNet_36.coatnet_0().to(device)
    
    #attack_set_all, wv = make_dataset_all(ATTACK_DATA_PATH, hop=50)
    attack_set_points, key_marker = make_dataset_points(ATTACK_DATA_PATH)
    #attack_loader_all    = torch.utils.data.DataLoader(attack_set_all,    batch_size=BATCH_SIZE, shuffle=False)
    attack_loader_points = torch.utils.data.DataLoader(attack_set_points, batch_size=BATCH_SIZE, shuffle=False)
    
    params = torch.load(MODEL_PATH,map_location=torch.device(device))
    net.load_state_dict(params)

    net.eval()
    
    # 連続で推論をかける
    """result = np.zeros(0)
    for i, (data,labels) in enumerate(attack_loader_all):
        data = data.to(device)
        labels = labels.to(device)
        output = net(data)
        output = output.detach().cpu().numpy().flatten()
        result = np.concatenate([result,output])
    result = result.flatten().reshape([-1,36])
    result = (np.array(result))"""
    
    # 打鍵部分だけ推論をかける
    # 1バッチで終わる前提
    keys = "0123456789abcdefghijklmnopqrstuvwxyz"
    for i, (data,labels) in enumerate(attack_loader_points):
        data = data.to(device)
        labels = labels.to(device)
        output = net(data)
        output_softmax = F.softmax(output, dim=1).cpu().detach().numpy()
        for j, out in enumerate(output_softmax):
            sorted = np.argsort(out)[::-1]
            print(f'key{j} : ', end='')
            for k in range(ATTACK_OUT_RANK):
                print(f'{keys[sorted[k]]} ({out[sorted[k]]*100}%), ', end='')
            print('')
        
    
    """df = pd.DataFrame(result, columns=['0','1','2','3','4','5','6','7','8','9',
                                       'a','b','c','d','e','f','g','h','i','j',
                                       'k','l','m','n','o','p','q','r','s','t',
                                       'u','v','w','x','y','z',])
    
    fig = make_subplots(rows=2, cols=1)
    wv = wv[0].numpy()
    fig.add_trace(go.Scatter(y=wv, mode='lines', name='sound'), row=1, col=1)
    fig.add_trace(go.Scatter(y=key_marker, mode='lines', name='key presence'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df['0'], mode='lines', name='0'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['1'], mode='lines', name='1'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['2'], mode='lines', name='2'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['3'], mode='lines', name='3'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['4'], mode='lines', name='4'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['5'], mode='lines', name='5'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['6'], mode='lines', name='6'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['7'], mode='lines', name='7'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['8'], mode='lines', name='8'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['9'], mode='lines', name='9'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['a'], mode='lines', name='a'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['b'], mode='lines', name='b'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['c'], mode='lines', name='c'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['d'], mode='lines', name='d'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['e'], mode='lines', name='e'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['f'], mode='lines', name='f'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['g'], mode='lines', name='g'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['h'], mode='lines', name='h'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['i'], mode='lines', name='i'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['j'], mode='lines', name='j'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['k'], mode='lines', name='k'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['l'], mode='lines', name='l'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['m'], mode='lines', name='m'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['n'], mode='lines', name='n'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['o'], mode='lines', name='o'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['p'], mode='lines', name='p'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['q'], mode='lines', name='q'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['r'], mode='lines', name='r'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['s'], mode='lines', name='s'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['t'], mode='lines', name='t'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['u'], mode='lines', name='u'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['v'], mode='lines', name='v'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['w'], mode='lines', name='w'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['x'], mode='lines', name='x'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['y'], mode='lines', name='y'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['z'], mode='lines', name='z'), row=2, col=1)
    #fig.add_trace(go.scatter(y=power,   mode='lines', name='p'), row=3, col=1)

    fig.update_yaxes(range=(0,np.max(result)), row=2, col=1)
    
    fig.write_html("plot.html", full_html=True, include_plotlyjs=True)"""
    
    return
 

if __name__ == '__main__': 
    args = sys.argv
    if len(args) != 2:
        print("incorrect arguments")
    elif args[1] == "learn":
        learn()
    elif args[1] == "test":
        test()
    elif args[1] == "attack":
        attack()
