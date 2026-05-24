import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


def visualize_audio(audio_path):
    print(f"\nLoading: {audio_path}")

    # Load audio
    # Load only first 30 seconds
    y, sr = librosa.load(audio_path, duration=30)

    print(f"Sample Rate: {sr}")
    print(f"Total Samples: {len(y)}")

    # -----------------------------------
    # 1. Waveform
    # -----------------------------------
    plt.figure(figsize=(14, 5))

    librosa.display.waveshow(y, sr=sr)

    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # 2. Spectrogram
    # -----------------------------------
    D = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                y,
                n_fft=1024,
                hop_length=512
            )
        ),
        ref=np.max
    )
    
    plt.figure(figsize=(12, 4))
    
    librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log"
    )
    
    plt.colorbar(format="%+2.0f dB")
    
    plt.title("Spectrogram")
    
    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # 3. Mel Spectrogram
    # -----------------------------------
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(14, 5))

    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Mel Spectrogram")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/visualize_audio.py path/to/song.mp3")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    if not audio_file.exists():
        print("File does not exist.")
        sys.exit(1)

    visualize_audio(audio_file)