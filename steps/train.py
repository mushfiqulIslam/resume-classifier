import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from steps.processor import process_string


def train(training_csv):
    df = pd.read_csv(training_csv)
    df = df.drop(['ID', 'Resume_html'], axis=1)

    df['processed_text'] = df['Resume_str'].apply(lambda w: process_string(w))
    df = df.drop(['Resume_str'], axis=1)

    vectorizer = CountVectorizer()
    vectorizer.fit(df['processed_text'])

    x_train, _, y_train, _ = train_test_split(df['processed_text'], df['Category'], test_size=0.2)

    transformed_train = vectorizer.transform(x_train).astype(float)

    model = OneVsRestClassifier(RandomForestClassifier(random_state=42, n_estimators=600, max_depth=12))
    model.fit(transformed_train, y_train)

    if not os.path.exists("bin"):
        os.mkdir("bin")

    with open('bin/vector.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

    with open('bin/classifier.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Training completed")
