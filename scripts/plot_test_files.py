import os
import matplotlib.pyplot as plt
import librosa

if __name__ == '__main__':
    test_dir = '../test/plc_challenge/enhanced/'
    os.makedirs('../test/plc_challenge/plots/', exist_ok=True)
    for i in range(7):
        filename = 'tgt_audio_' + str(i) + '.wav'
        matches = [
            test_dir + 'naive/2.0.1/' + filename,
            test_dir + 'resConnections/3.0.1/' + filename,
            test_dir + 'outputRNN/4.0.1/' + filename,
        ]
        plt.figure()
        for match in matches:
            version = match.split('/')[-2]
            y, sr = librosa.load(match)
            plt.plot(y, label=version)
        plt.title(filename)
        plt.legend()
        plt.show()
        plt.savefig('../test/plc_challenge/plots/' + filename[:-4] + '.png')
