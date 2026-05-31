from app.core.storage import upload_file_to_s3
from pathlib import Path

local_file = Path("data/test_song.mp3")
s3_key = "test/test_song.mp3"

upload_file_to_s3(local_file, s3_key)
print("Upload successful!")
