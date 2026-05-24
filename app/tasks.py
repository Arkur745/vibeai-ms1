from app.celery_app import celery
from app.inference import predict_emotion

import traceback


@celery.task
def process_audio_task(audio_path):

    try:

        print("\n====================")
        print("TASK STARTED")
        print("====================")

        print(f"Audio Path: {audio_path}")

        prediction = predict_emotion(audio_path)

        print("\nPrediction Success")
        print(prediction)

        return prediction

    except Exception as e:

        print("\nTASK FAILED")
        print(str(e))

        traceback.print_exc()

        raise e
