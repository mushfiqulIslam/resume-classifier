import os

from dotenv import load_dotenv

from steps import train
from steps.classify import classify_resume

if __name__ == '__main__':
    load_dotenv()
    train_first = os.getenv("TRAIN_FIRST", default=False) in ('true', '1', 't')
    if train_first:
        training_csv = os.getenv("TRAINING_CSV", default=False)
        if not training_csv:
            print("Please provide csv file url on env file")
            exit(1)

        print("Started training")
        train(training_csv=training_csv)
        print("Training finished")

    data_path = os.getenv("DATA_PATH_TO_CLASSIFY", default=False)
    if data_path:
        classify_resume(data_path)
    else:
        print("No data provided to classify. Exiting")
