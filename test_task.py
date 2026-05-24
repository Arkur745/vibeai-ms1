from app.tasks import process_audio_task


result = process_audio_task.delay(
    r"C:\dev\SAAS\vibeai-ms1\data\test_song.mp3"
)

print("Task Submitted")

print("Task ID:", result.id)

print("Waiting for result...")

print(result.get(timeout=120))
