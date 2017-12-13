from pydub import AudioSegment
import os



def detect_leading_silence(sound, silence_threshold=-28.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms

for x in ["go", "left", "right", "stop"]:
    for y in range(1,21):
        print("1")
        sound = AudioSegment.from_file(x+"_"+str(y)+".wav", format="wav")
        print("2")
        start_trim = detect_leading_silence(sound)
        print(start_trim)
        print("3")
        end_trim = detect_leading_silence(sound.reverse())
        print("4")
        print(end_trim)
        duration = len(sound)    
        trimmed_sound = sound[start_trim-60:duration-end_trim+60]
        print("5")
        trimmed_sound.export(x+"_"+str(y)+".wav", format="wav")
        print("done")
        