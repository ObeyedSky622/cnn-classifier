import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import chirp
from scipy.signal import stft


def main(path, noise=False):
    """

    """
    print(f"saving dataset to {path} from {os.getcwd()}")
    np.random.RandomState(42)
    inits = [250, 500, 750, 1000, 500, 500, 1000, 2000, 5000, 2250]
    fins = [1000, 1500, 3000, 4000, 5000, 0, 0, 500, 750, 320]
    for i in range(50):

        # Time array of 1-5 micro seconds
        j = 0
        for init_freq, fin_freq in zip(inits, fins):

            # bring freq to MHz range
            init_freq *= 1E6
            fin_freq *= 1E6

            # set up time scale
            fs = 10000
            delta_t_int = 1
            delta_t = delta_t_int * 1E-6
            x = np.linspace(0, delta_t, int(delta_t_int*fs))
            yl = chirp(x, init_freq, delta_t, fin_freq, method='linear') + \
                np.random.normal(0, 1, len(x))

            yq = chirp(x, init_freq, delta_t, fin_freq, method='quadratic') + \
                np.random.normal(0, 1, len(x))

            plt.figure()
            plt.specgram(yl, Fs=fs, Fc=init_freq, NFFT=256)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(path + '/lin_'+str(i)+'_'+str(j)+'.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            plt.figure()
            plt.specgram(yq, Fs=fs, Fc=init_freq, NFFT=256)
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(path + '/quad_'+str(i)+'_'+str(j)+'.png',
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            j += 1

    print("Dataset Complete")


if __name__ == "__main__":
    main("data/", False)
