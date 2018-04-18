from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
import IPython.display

import librosa
import librosa.display

audio_path = '/home/cdsw/7061-6-0-0.wav'

y, sr = librosa.load(audio_path)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
    fig, ax1 = plt.subplots(1,1)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    # Make a new figure
    #plt.figure(figsize=(12,4))
    plt.title('mel power spectrogram')
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+02.0f dB')

import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  y_harmonic, y_percussive = librosa.effects.hpss(y)
  # What do the spectrograms look like?
  # Let's make and display a mel-scaled power (energy-squared) spectrogram
  S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
  S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)
  # Convert to log scale (dB). We'll use the peak power as reference.
  log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
  log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
  # Make a new figure
  plt.figure(figsize=(12,6))
  plt.subplot(2,1,1)
  # Display the spectrogram on a mel scale
  librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')
  # Put a descriptive title on the plot
  plt.title('mel power spectrogram (Harmonic)')
  # draw a color bar
  plt.colorbar(format='%+02.0f dB')
  plt.subplot(2,1,2)
  librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')
  # Put a descriptive title on the plot
  plt.title('mel power spectrogram (Percussive)')
  # draw a color bar
  plt.colorbar(format='%+02.0f dB')

## Chromogram
# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients

import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
  # Make a new figure
  plt.figure(figsize=(12,4))
  # Display the chromagram: the energy in each chromatic pitch class as a function of time
  # To make sure that the colors span the full range of chroma values, set vmin and vmax
  librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
  plt.title('Chromagram')
  plt.colorbar()

# Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)
# Let's pad on the first and second deltas while we're at it
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

## MFCC
import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  # How do they look?  We'll show each in its own subplot
  plt.figure(figsize=(12, 6))

  plt.subplot(3,1,1)
  librosa.display.specshow(mfcc)
  plt.ylabel('MFCC')
  plt.colorbar()

  plt.subplot(3,1,2)
  librosa.display.specshow(delta_mfcc)
  plt.ylabel('MFCC-$\Delta$')
  plt.colorbar()

  plt.subplot(3,1,3)
  librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
  plt.ylabel('MFCC-$\Delta^2$')
  plt.colorbar()

##Beat Tracking
import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  # Now, let's run the beat tracker.
  # We'll use the percussive component for this part
  plt.figure(figsize=(12, 6))
  tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
  # Let's re-draw the spectrogram, but this time, overlay the detected beats
  plt.figure(figsize=(12,4))
  librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
  # Let's draw transparent lines over the beat frames
  plt.vlines(librosa.frames_to_time(beats),
             1, 0.5 * sr,
             colors='w', linestyles='-', linewidth=2, alpha=0.5)
  plt.axis('tight')
  plt.colorbar(format='%+02.0f dB')

## Beat-synchronous feature aggregation
import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  # feature.sync will summarize each beat event by the mean feature vector within that beat
  M_sync = librosa.util.sync(M, beats)
  plt.figure(figsize=(12,6))
  # Let's plot the original and beat-synchronous features against each other
  plt.subplot(2,1,1)
  librosa.display.specshow(M)
  plt.title('MFCC-$\Delta$-$\Delta^2$')
  # We can also use pyplot *ticks directly
  # Let's mark off the raw MFCC and the delta features
  plt.yticks(np.arange(0, M.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])
  plt.colorbar()
  plt.subplot(2,1,2)
  # librosa can generate axis ticks from arbitrary timestamps and beat events also
  librosa.display.specshow(M_sync, x_axis='time',
                           x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))
  plt.yticks(np.arange(0, M_sync.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])             
  plt.title('Beat-synchronous MFCC-$\Delta$-$\Delta^2$')
  plt.colorbar()
  
  ## Beat-synchronous feature aggregation
import matplotlib.pyplot as plt
with plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 'ytick.color':'green', 'figure.facecolor':'white', "axes.labelcolor": "green"}):
  C_sync = librosa.util.sync(C, beats, aggregate=np.median)

  plt.figure(figsize=(12,6))

  plt.subplot(2, 1, 1)
  librosa.display.specshow(C, sr=sr, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time')

  plt.title('Chroma')
  plt.colorbar()

  plt.subplot(2, 1, 2)
  librosa.display.specshow(C_sync, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time', 
                           x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))


  plt.title('Beat-synchronous Chroma (median aggregation)')

  plt.colorbar()
