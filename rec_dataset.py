
import keyboard
import pyaudio
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Value, Process


FORMAT = pyaudio.paInt16
RATE = 44100
CHANNEL = 1
FILEPATH = 'dataset/test'
frame_size = 1024
frame_num  = 256

def audiostart(buf_num):
    audio = pyaudio.PyAudio()
    stream = audio.open( format=FORMAT, 
                         rate=RATE, 
                         channels=CHANNEL, 
                         input_device_index=1, 
                         input=True, 
                         frames_per_buffer=buf_num
                        )
    return audio, stream

def audiostop(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()

def max_index_exclude_range(arr, a, b):
    """
    配列 arr のうち、インデックス a 以上 b 未満の区間を除いた部分で最大値のインデックスを返す。
    （a > b の場合は自動的に入れ替え）
    """
    # a, b の順を自動調整
    if a > b:
        a, b = b, a

    # 範囲チェック
    if a < 0 or b > len(arr):
        raise ValueError("区間 [a, b) が不正です")

    # 除外部分をマスクで無効化
    mask = np.ones_like(arr, dtype=bool)
    mask[a:b] = False

    # 最大値のインデックス（元の配列での位置）
    max_idx = np.argmax(arr[mask])
    
    # 元のインデックスに変換
    original_indices = np.where(mask)[0]
    return original_indices[max_idx]

def align_phrase_to_peak(phrase, align_center=5000):
    abs_phrase = np.abs(phrase)
    peak_idx_a = np.argmax(abs_phrase)
    peak_idx_b = max_index_exclude_range(abs_phrase, peak_idx_a-800, peak_idx_a+800)
    peak_idx = min(peak_idx_a, peak_idx_b)

    phrase = phrase[peak_idx-align_center:peak_idx+16384-align_center]

    return phrase


# -------------------------------------------------------------

def key_detect(sm):
    while True:
        key = keyboard.read_event(suppress=False)
        key_name = key.name
        if key.event_type == 'down':
            if len(key_name) == 1:
                #print(f'key : {key_name} was detected.')
                time.sleep(frame_size*20/RATE)   # 遅らせて送信
                sm.value = key_name
            if key.name=='/': break
    return

def save(sm):
    audio, stream = audiostart(frame_size)
    sound_queue = np.zeros((frame_num,frame_size),dtype='int16')
    index = 0
    sound_id = 0

    print(f'Queue shape : {sound_queue.shape}')
    print("Recoding Start...")


    while True:
        # キーの読み取り
        # キーを読み込んだら初期化文字を代入．
        key = sm.value
        sm.value = '*'  

        # キューに1フレーム分を代入
        sound_queue[index] = np.frombuffer(stream.read(frame_size), dtype='int16')
        #print(f'index : {index}')
        

        if key == '/' : 
            break

        if key != '*':
            # フレーズを多めに抽出する
            index_from = np.mod(index-25, frame_num)
            index_to = np.mod(index,frame_num)
            if index_from < index_to:
                phrase = sound_queue[index_from:index_to].flatten()
            else:
                phrase = np.concatenate([sound_queue[index_from:], sound_queue[:index_to]]).flatten()
                
            # 多めに抽出したフレーズをピーク値でカット
            peaked_phrase = align_phrase_to_peak(phrase)
            
            #print(phrase.shape)
            #plt.plot(peaked_phrase)
            #plt.show()
            # 保存
            fn = FILEPATH + '/' + key + '_' + str(sound_id) + '.wav'
            wavFile = wave.open(fn, 'wb')
            #wavFile = wave.open('testsound.wav', 'wb')
            wavFile.setnchannels(CHANNEL)
            wavFile.setsampwidth(audio.get_sample_size(FORMAT))
            wavFile.setframerate(RATE)
            wavFile.writeframes(b"".join(peaked_phrase))
            wavFile.close() 
            print(f"Phrase {key} was saved. File name is : {fn}")
            
            sound_id += 1

        index = np.mod(index+1,frame_num)
        #

    audiostop(audio, stream)
    
    return



def main():
    sm = Value('u', '*')

    print("Multiprocess start.")

    key_detect_process = Process(target=key_detect, args=(sm,))
    stft_process = Process(target=save, args=(sm,))

    key_detect_process.start()
    stft_process.start()

    key_detect_process.join()
    stft_process.join()

    print("All processes ended. ")

    return

if __name__ == '__main__':
    main()



