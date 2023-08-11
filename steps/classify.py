import os
import pickle

import pandas as pd
from PyPDF2 import PdfReader

from steps.processor import process_string


def classify_resume(data_path):
    pdf_list = []
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            pdf_list.append(os.path.join(dirname, filename))

    df = pd.DataFrame({
        'resume_url': pdf_list,
    })

    resume_text_list = []
    for i in df.index:
        with open(df['resume_url'][i], 'rb') as file:
            reader = PdfReader(file)
            page_count = len(reader.pages)
            text = ""
            for page in range(page_count):
                if page != 0:
                    text += " "

                text += reader.pages[page].extract_text()

        resume_text_list.append(text)

    df["text"] = resume_text_list
    df["processed_text"] = df["text"].apply(lambda w: process_string(w))
    df = df.drop(["text"], axis=1)

    with open('bin/vector.pkl', 'rb') as file:
        # Load the object from the file
        vectorizer = pickle.load(file)

    transformed_text = vectorizer.transform(df['processed_text']).astype(float)

    with open('bin/classifier.pkl', 'rb') as file:
        model = pickle.load(file)

    pred = model.predict(transformed_text)
    df = df.drop(["processed_text"], axis=1)
    df["prediction"] = pred

    for i in df.index:
        prev_path = df['resume_url'][i]
        new_path = os.path.join(data_path, df["prediction"][i])
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        file_name = prev_path.replace(data_path, "").replace("\\", "")
        os.rename(prev_path, os.path.join(new_path, file_name))

    df.to_csv(os.path.join(data_path, "categorized_resumes.csv"))
    print("Done")


