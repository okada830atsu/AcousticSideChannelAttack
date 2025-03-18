
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
FILEPATH = 'attack_files/helloworld0319.wav'
frame_size = 1024

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

# -------------------------------------------------------------

def key_detect(sm):
    while True:
        key = keyboard.read_event(suppress=False)
        key_name = key.name
        if key.event_type == 'down':
            if key_name == '/':
                sm.value = key_name
                break
    return

def savesounds(sm):
    audio, stream = audiostart(frame_size)
    sound = []
    
    time.sleep(1)
    print("Recoding Start...")

    while True:
        # キーの読み取り
        key = sm.value
        # キューに1フレーム分を代入
        sound.append(np.frombuffer(stream.read(frame_size), dtype='int16'))

        if key == '/' :  
            sound = np.array(sound).flatten()[3000:]
            # 保存
            wavFile = wave.open(FILEPATH, 'wb')
            wavFile.setnchannels(CHANNEL)
            wavFile.setsampwidth(audio.get_sample_size(FORMAT))
            wavFile.setframerate(RATE)
            wavFile.writeframes(b"".join(sound))
            wavFile.close() 
            audiostop(audio, stream)
            print("Recording completed!")
            plt.plot(sound)
            plt.show()
            print(f"Saved to {FILEPATH}")
            break
    return



def main():
    sm = Value('u', '*')

    print("Multiprocess start.")

    key_detect_process = Process(target=key_detect, args=(sm,))
    stft_process = Process(target=savesounds, args=(sm,))

    key_detect_process.start()
    stft_process.start()

    key_detect_process.join()
    stft_process.join()

    print("All processes ended. ")

    return

if __name__ == '__main__':
    main()



