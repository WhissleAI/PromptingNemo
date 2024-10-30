import sys
import glob
import os
from pydub import AudioSegment


from speechbrain.inference.interfaces import foreign_class
emotion_classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

label_mapping = {
    'neu': 'Neutral',
    'hap': 'Happy',
    'ang': 'Angry',
    'sad': 'Sad'
}


def split_and_emotion(input_tsv, audio_file, output_tsv, output_folder, fileid):
    '''
    0.0	1.36	Doctor	What brought you in today? INTENT_GREETING

    1.36	10.48	Patient	Sure, I'm I'm just having a lot of ENTITY_SYMPTOM chest pain END and and so I thought I should get it checked out. INTENT_SEEK_MEDICAL_HELP

    11.2	20.08	Doctor	OK, before we start, could you remind me of your ENTITY_GENDER gender END and ENTITY_AGE age END? INTENT_INQUIRE_DEMOGRAPHIC_INFO
    '''

    input_lines = open(input_tsv).readlines()
    output_tsv = open(output_tsv, 'w')
    
    audio = AudioSegment.from_wav(audio_file)    

    real_num = 0
    for num in range(0,len(input_lines)):
        
        line1 = input_lines[num]
        
        
        #print(line)
        if line1.strip() != '' and num < len(input_lines) - 2:
            
            line2 = input_lines[num+2]
            
            #print("Line1", line1)
            #print("Line2", line2)
            
            
            output_audio_file = fileid + "_" + str(real_num) + "_" + str(real_num+1) + ".wav"
            output_audio_file = os.path.join(output_folder, output_audio_file)
            #print("Output audio file", output_audio_file)
            line1 = line1.strip().split("\t")
            line2 = line2.strip().split("\t")

            start_time1, end_time1, role1, text1 = line1
            
            role1 = "ROLE_"+role1.upper()
            #print("\n\n Line 1 details\n")
            #print(start_time1, end_time1, role1, text1)
            
            start1 = float(start_time1) * 1000
            end1 = float(end_time1) * 1000
            
            start_time2, end_time2, role2, text2 = line2
            role2 = "ROLE_"+role2.upper()
            #print("\n\n Line 2 details\n")
            #print(start_time1, end_time1, role1, text1)
            
            start2 = float(start_time2) * 1000
            end2 = float(end_time2) * 1000
            
            start = start1  # Start time in milliseconds
            end = end2    # End time in milliseconds
            
            sliced_combined_audio = audio[start:end]
            
            sliced_audio = audio[start1:end1]
            sliced_audio.export("output.wav", format="wav")
            out_prob, score, index, text_lab1 = emotion_classifier.classify_file("output.wav")
            
            sliced_audio = audio[start2:end2]
            sliced_audio.export("output.wav", format="wav")
            out_prob, score, index, text_lab2 = emotion_classifier.classify_file("output.wav")
            
            
            emotion_label1 = "EMOTION_" + label_mapping[text_lab1[0]].upper()
            emotion_label2 = "EMOTION_" + label_mapping[text_lab2[0]].upper()
            
            output_tsv.write(output_audio_file + "\t" + text1 + " " + role1 + " " + emotion_label1 + " TURN_CHANGE " + text2 + " " + role2 + " " + emotion_label2 + "\n")

            sliced_combined_audio.export(output_audio_file, format="wav")
            
            real_num = real_num + 1        
 
    output_tsv.close()
    
    
    
    

if __name__ == '__main__':
    
    input_folder = sys.argv[1]
    audio_folder = sys.argv[2]
    output_folder = sys.argv[3]
    
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        
        input_tsv_file = os.path.join(input_folder, filename)
        
        
        output_tsv_file = os.path.join(output_folder, filename.replace('.tsv', '_emotion.tsv'))
        
        #output_tsv_file = os.path.join(output_folder, filename.replace('.tsv', '_annotated.tsv'))
        
        fileid =  filename.split("_")[0]
        audio_file = os.path.join(audio_folder, fileid + '.wav')

        split_and_emotion(input_tsv_file, audio_file, output_tsv_file, output_folder, fileid)
        
