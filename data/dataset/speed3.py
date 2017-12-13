from pydub import AudioSegment


def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

for x in ["go", "left", "right", "stop"]:
    for y in range(1,21):
		for x1 in range(75,125):
			sound = AudioSegment.from_file(x+"_"+str(y)+".wav", format="wav")
			temp = x1/100.0
			soundnew = speed_change(sound, temp)
			soundnew.export(x+"_"+str(y)+"_"+str(x1)+".wav", format="wav")