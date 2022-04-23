import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import soundfile as sf

FORMAT = pyaudio.paInt16 
CHANNELS = 1
FS = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "file.wav"



def process_data(in_data, frame_count, time_info, flag):
    global Frame_buffer,frame_idx, delay
    in_audio_data = np.frombuffer(in_data, dtype=np.int16)
    Frame_buffer[frame_idx:(frame_idx+CHUNK),0]=in_audio_data
    ################################
    ## Do something wih data
    delay[0:CHUNK] = np.copy(in_audio_data)
    delay = np.roll(delay, -CHUNK)
    out_audio_data = np.copy(delay[0:CHUNK])
    ################################

    Frame_buffer[frame_idx:(frame_idx+CHUNK),1]=out_audio_data
    out_data = out_audio_data.tobytes()
    frame_idx += CHUNK
    return out_data, pyaudio.paContinue

def save_video(output):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter("original.avi", fourcc, 30.0, (1280,  720))

    fourcc2 = cv2.VideoWriter_fourcc(*'X264')
    out2 = cv2.VideoWriter("processed.avi", fourcc2, 30.0, (200,  200))

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Display the resulting frame
        frame_flip = cv2.flip(frame, 180)
        frame_flipclip = frame_flip[200:400,400:600].copy()
        cv2.imshow('frame', frame)
        
        out.write(frame)
        out2.write(frame_flipclip)
        
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    out.release()
    cap.release()
    cv2.destroyAllWindows()

save_video("kamera.avi")

def check_audio_devices():
    audio = pyaudio.PyAudio()
    numdevices = audio.get_device_count()
    # for i in range(0, numdevices):
        # print(audio.get_device_info_by_index(i))
    return audio

audio = check_audio_devices()

stream = audio.open(input_device_index =1,
                    output_device_index=3,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=process_data
                    )

global Frame_buffer,frame_idx, delay
N=10
frame_idx=0
Frame_buffer = np.zeros(((N+1)*FS,2))
delay_len = 0.2
delay_size = (CHUNK * int(FS / CHUNK * delay_len))
delay = np.zeros( delay_size, dtype=np.int16)

stream.start_stream()
while stream.is_active():
    time.sleep(N)
    stream.stop_stream()
stream.close()

## zapis do pliku
sf.write('nazwa.wav', Frame_buffer.astype(np.int16), FS)

ox = np.arange(0, len(Frame_buffer))
ox = ox / FS

plt.subplot(2,1,1)
plt.plot(ox,Frame_buffer[:,0])
plt.subplot(2,1,2)
plt.plot(ox,Frame_buffer[:,1])
plt.show()
