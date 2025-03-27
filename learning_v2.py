

import U_Net_37
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import os
import re
import glob
import sys
import random

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

SAMPLE_RATE         = 44100
N_MEL               = 64
N_FFT               = 1024
N_HOP               = 192
TRAIN_DATA_PATH     = 'dataset/train'
VAL_DATA_PATH       = 'dataset/val'
TEST_DATA_PATH      = 'dataset/test'
MODEL_PATH          = 'models/model_4.pt'
ATTACK_DATA_PATH    = 'attack_files/helloworld0323.wav'
AUG_MASK            = 1
BATCH_SIZE          = 64
N_EPOCH             = 200
LR                  = 0.001
ATTACK_OUT_RANK     = 4




def key_to_label_val(key):

    keys = "0123456789abcdefghijklmnopqrstuvwxyz_"
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

    return spec

def apply_spec_augment(spectrogram, time_mask_param=10, freq_mask_param=8, num_time_masks=2, num_freq_masks=2):
    """
    spectrogram: Tensor of shape (1, H=mel_bands, W=frames)
    """
    augmented = spectrogram.clone()
    _, H, W = augmented.shape

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(1, W - t))
        augmented[:, :, t0:t0 + t] = augmented.mean()

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(1, H - f))
        augmented[:, f0:f0 + f, :] = augmented.mean()

    return augmented

def label_mask(n,val,a,b):
    """
    長さnのラベルで，区間[a,b]がラベル値valになる.
    それ以外の区間は打鍵なし判定の37を割り当てる．
    """
    label = np.ones(n) * 36
    for i in range(a,b,1):
        label[i] = val
    return label

def make_dataset(path, num_mask):
    
    files = glob.glob(path+'/*')
    num = len(files)
    
    X = torch.zeros(num*num_mask,1,64,64, dtype=torch.float32)
    y = torch.zeros(num*num_mask,64, dtype=torch.int64)
    
    
    for filecount,file in enumerate(tqdm(files)):
        #print(file)
        waveform, sample_rate = torchaudio.load(uri=file)
        
        spec = get_melspectrogram(waveform)[:,11:11+64] #中央部分をカット
        
        label_val = re.search(r'\\(.+).',file).group(1)    # \ と \ に挟まれた部分を検索し，int型に変換
        label_val = key_to_label_val(str(label_val[0]))
        label = label_mask(64, label_val, 4, 56)
        
        #plt.plot(waveform)
        #plt.imshow(spec)
        #plt.plot(label, c='white')
        #plt.show()

        X[filecount] = torch.from_numpy(spec)
        y[filecount] = torch.from_numpy(label)
    

    dataset = torch.utils.data.TensorDataset(X,y)
    
    return dataset

def make_dataset_all(path):
     
    waveform, sample_rate = torchaudio.load(uri=path)
    
    spec= get_melspectrogram(waveform)
    
    plt.imshow(spec)
    plt.show()
    
    X = torch.zeros(1,1,64,spec.shape[1], dtype=torch.float32)
    y = torch.zeros(1,spec.shape[1], dtype=torch.int64)
    
    X[0] = torch.from_numpy(spec)
    #y[0] = torch.zeros() # 適当な数字
        
    dataset = torch.utils.data.TensorDataset(X,y)
    
    return dataset, waveform, spec


def learn():
    train_set = make_dataset(TRAIN_DATA_PATH, AUG_MASK)
    val_set   = make_dataset(VAL_DATA_PATH, AUG_MASK)
    print("Datasets were created.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    history = {"train_loss": [], "validation_loss": [], "train_acc": [], "validation_acc": []}

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax_loss, ax_acc = ax[0], ax[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')

    net = U_Net_37.UNet1DSlice().to(device)
    print(net)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
    loss_func = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    print("Start...")
    for e in range(N_EPOCH):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for data, labels in tqdm(train_loader):
            # --- オンラインデータオーギュメンテーションをここで適用 ---
            #data_aug = torch.zeros_like(data)
            #for i in range(data.shape[0]):
            #    data_aug[i] = apply_spec_augment(data[i])
            #data = data_aug
            # -----------------------------------------------------------
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(data)  # (B, 64, 37)
            loss = loss_func(output.permute(0, 2, 1), labels)
            train_loss += loss.item()
            pred = output.argmax(dim=2)  # (B, 64)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        accuracy_train = correct / total

        #scheduler.step()

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                output = net(data)
                loss = loss_func(output.permute(0, 2, 1), labels)
                val_loss += loss.item()
                pred = output.argmax(dim=2)
                correct += (pred == labels).sum().item()
                total += labels.numel()

        val_loss /= len(val_loader)
        accuracy_val = correct / total

        if accuracy_val > best_accuracy:
            torch.save(net.state_dict(), MODEL_PATH)
            best_accuracy = accuracy_val

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["train_acc"].append(accuracy_train)
        history["validation_acc"].append(accuracy_val)

        print(f"Epoch {e}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={accuracy_train:.4f}, Val Acc={accuracy_val:.4f}, Best={best_accuracy:.4f}")

        ax_loss.cla()
        ax_loss.plot(history["train_loss"], label="train loss")
        ax_loss.plot(history["validation_loss"], label="val loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()

        ax_acc.cla()
        ax_acc.plot(history["train_acc"], label="train acc")
        ax_acc.plot(history["validation_acc"], label="val acc")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Epoch")
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
    plt.plot(range(1, N_EPOCH+1), history["train_acc"])
    plt.plot(range(1, N_EPOCH+1), history["validation_acc"])
    plt.title("test accuracy")
    plt.xlabel("epoch")
    plt.savefig("img/test_acc.png")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')

    net = U_Net_37.UNet1DSlice().to(device)
    test_set = make_dataset(TEST_DATA_PATH, AUG_MASK)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    params = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(params)
    net.eval()

    correct = 0
    total = 0

    all_preds = []  # 全ての予測
    all_labels = []  # 全ての正解

    for i, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        pred = output.argmax(dim=2)  # (B,64)

        correct += (pred == labels).sum().item()
        total += labels.numel()

        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    acc = correct / total
    print("Test Accuracy: {:.4f}".format(acc))

    # --- 混同行列の描画 ---
    all_preds = np.concatenate(all_preds).flatten()   # (B*64,)
    all_labels = np.concatenate(all_labels).flatten() # (B*64,)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(37)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)] + list("abcdefghijklmnopqrstuvwxyz") + ["none"])
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return


def attack():
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device : {device}')
    
    net = U_Net_37.UNet1DSlice().to(device)
    
    attack_set, wv, spec = make_dataset_all(ATTACK_DATA_PATH)
    attack_loader = torch.utils.data.DataLoader(attack_set, batch_size=BATCH_SIZE, shuffle=False)
    
    params = torch.load(MODEL_PATH,map_location=torch.device(device))
    net.load_state_dict(params)

    net.eval()
    
    # 連続で推論をかける
    output = None
    for i, (data,labels) in enumerate(attack_loader):
        data = data.to(device)
        labels = labels.to(device)
        output = net(data)
        output_softmax = F.softmax(output, dim=2).cpu().detach().numpy()
    
    df = pd.DataFrame(output_softmax[0], columns=['0','1','2','3','4','5','6','7','8','9',
                                       'a','b','c','d','e','f','g','h','i','j',
                                       'k','l','m','n','o','p','q','r','s','t',
                                       'u','v','w','x','y','z','_'])
    
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05)

    # 画像を追加（imshowではなくImage traceを使う）
    fig.add_trace(go.Heatmap(z=spec,colorscale='Viridis',showscale=False),row=1, col=1)
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
    fig.add_trace(go.Scatter(y=df['_'], mode='lines', name='None'), row=2, col=1)
    #fig.add_trace(go.scatter(y=power,   mode='lines', name='p'), row=3, col=1)

    fig.update_yaxes(range=(0,np.max(output_softmax)), row=2, col=1)
    
    fig.write_html("plot.html", full_html=True, include_plotlyjs=True)
    
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
