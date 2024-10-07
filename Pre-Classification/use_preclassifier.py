from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import numpy as np
import pandas as pd
import os

# load data

print('Loading data...')

data = []

app = "whatsapp"

with open(f"H:\Laptop\last_3yeras_50k\{app}_results_splitted_corrected.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        if (len(line.strip()) > 0):
            data.append(line.strip())

print("Data loaded")

texts = data

# Define file paths
model_path = 'trained_logistic_regression_updated.pkl'

# Check if resuming training and load the saved classifier if needed
model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = joblib.load(model_path)

# Define the categories (labels) used during training
categories = ['BUG', 'FEATURE', 'USE', 'DISLIKE_REASON', 'LIKE_REASON', 'SIMPLE_RECOMMENDATION', 'SIMPLE_LIKE', 'SIMPLE_DISLIKE']
id2label = {0: "BUG", 1: "FEATURE", 2: "USE", 3: "DISLIKE_REASON", 4: "LIKE_REASON", 5: 'SIMPLE_RECOMMENDATION', 6: 'SIMPLE_LIKE', 7: 'SIMPLE_DISLIKE'}
label2id = {"BUG": 0, "FEATURE": 1, "USE": 2, "DISLIKE_REASON": 3, "LIKE_REASON": 4, 'SIMPLE_RECOMMENDATION': 5, 'SIMPLE_LIKE': 6, 'SIMPLE_DISLIKE': 7}

results = []
batch_size = 1000

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]

    # Generate embeddings for training data
    unlabeled_embeddings = np.array(model.encode(batch_texts, show_progress_bar=True)).reshape(-1, 384)

    # Predict probabilities for the unlabeled data
    y_pred_proba = classifier.predict_proba(unlabeled_embeddings)

    # Convert the label indices to the corresponding categories and get their probabilities
    for idx, probs in enumerate(y_pred_proba):
        predicted_label = categories[np.argmax(probs)]
        probability = max(probs)
        results.append((batch_texts[idx], predicted_label, probability))

    print(f"Processed: {i + batch_size}/{len(texts)}")

print("Label sequences generated")

print("Saving results...")

sorted_data_by_group = sorted(results, key=lambda x: (x[1], -x[2]))

probability_threshold = 0.70

with open(f"results_grouped_by_label_classifier_{app}.txt", "w", encoding="utf-8") as f:
    first = True
    change = ""
    change_prev = change

    threshold = True
    threshold_first = True

    others = []

    for idx, (text, label, probability) in enumerate(sorted_data_by_group):
        if idx % 10 == 0 and idx != 0:
            print(f"Processed: {idx}/{len(sorted_data_by_group)}")

        change_prev = change
        change = label

        if change != change_prev:
            first = True
            threshold = True
            threshold_first = True

        if probability < probability_threshold:
            threshold = True

        if first:
            f.write("---------------------------------------------------------------\n")
            f.write(f"{label}:\n")
            f.write('\n\n')
            first = False

        if threshold and probability > probability_threshold:
            f.write("Confident (Above probability_threshold)\n")
            threshold = False
        elif threshold and probability < probability_threshold and threshold_first:
            f.write("\nNot Confident (Below probability_threshold)\n\n")
            threshold = False
            threshold_first = False

        if probability >= probability_threshold:
            f.write(f"{text} | {label} | Confidence: {probability}\n")
        else:
            others.append((text, label, probability))

    f.write("---------------------------------------------------------------\n")
    f.write("Others:\n")
    f.write('\n\n')

    for idx, (text, label, probability) in enumerate(others):
        f.write(f"{text} | {label} | Confidence: {probability}\n")

df = pd.DataFrame()

df['text'] = texts
df['label'] = [label2id[label] for text, label, prob in results]
df['probs'] = [prob for text, label, prob in results]

df.to_csv(f"H:\Laptop\last_3yeras_50k\{app}_classifier.csv", index=False, encoding='utf-8')

print("Results saved")