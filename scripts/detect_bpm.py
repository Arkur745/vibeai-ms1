import librosa
import sys
from pathlib import Path
import mlflow


def detect_bpm(audio_path):
    print(f"\nLoading file: {audio_path}")

    # Start MLflow run
    with mlflow.start_run():

        # Load audio
        y, sr = librosa.load(audio_path)

        duration = librosa.get_duration(y=y, sr=sr)

        print(f"Sample Rate: {sr}")
        print(f"Duration: {duration:.2f} seconds")

        # BPM detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

        # Normalize common double-tempo issue
        if tempo > 180:
            tempo = tempo / 2

        print(f"\nDetected BPM: {tempo.item():.2f}")

        # Log parameters
        mlflow.log_param("audio_file", str(audio_path))
        mlflow.log_param("sample_rate", sr)

        # Log metrics
        mlflow.log_metric("duration_seconds", duration)
        mlflow.log_metric("detected_bpm", float(tempo))

        print("\nMLflow tracking complete.")

        return tempo


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python scripts/detect_bpm.py path/to/song.mp3")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    if not audio_file.exists():
        print("File does not exist.")
        sys.exit(1)

    detect_bpm(audio_file)