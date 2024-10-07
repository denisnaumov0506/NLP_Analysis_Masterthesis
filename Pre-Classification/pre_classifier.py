from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import numpy as np
import pandas as pd
import os

# Define file paths
initial_dataset_path = "SequenceClassification/text_classification_final_splitted.txt"
unseen_path = "SequenceClassification/unseen.txt"
seen_path = "SequenceClassification/seen.csv"
model_path = 'trained_logistic_regression_updated.pkl'

cache = []

result_string = ""

resume_train = False

if os.path.exists(seen_path):
    seen_df = pd.read_csv(seen_path, encoding="utf-8")

# Load the dataset
if os.path.exists(unseen_path):
    load_unseen = input(f"Found '{unseen_path}'. Do you want to resume training from it? (yes/no): ")
    if load_unseen.lower() == 'yes':
        resume_train = True
        with open(unseen_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            texts = [line.strip() for line in lines]
            df = pd.DataFrame({'text': texts, 'label': [None] * len(texts)})
    else:
        with open(initial_dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            texts = [line.strip() for line in lines]
            df = pd.DataFrame({'text': texts, 'label': [None] * len(texts)})
else:
    with open(initial_dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        texts = [line.strip() for line in lines]
        df = pd.DataFrame({'text': texts, 'label': [None] * len(texts)})

# Separate labeled and unlabeled data
labeled_df = df.dropna(subset=['label'])
unlabeled_df = df[df['label'].isna()]

# Manually provide one example for each of the 8 labels if not resuming
if not resume_train:
    X_train = [
        "App is crashing regularly",                                                                                    # BUG (0)
        "App keep on crashing",                                                                                         # BUG (0)
        "It kepps on crashing",                                                                                         # BUG (0)
        "Live video does not work on my phone",                                                                         # BUG (0)
        "Lagging and delayed typing.",                                                                                  # BUG (0)
        "just please please please please add offline mod",                                                             # FEATURE (1)
        "Please add one more features in which we can collect gold, elixiar or dark elixiar separatly from treasury.",  # FEATURE (1)
        "Please add some bank for payment..",                                                                           # FEATURE (1)
        "MS, please add the option to delete emails from notification.",                                                # FEATURE (1)
        ", and please please please add the nether...",                                                                 # FEATURE (1)
        "Keeps me on task with my goal of reading the entire Bible.",                                                   # USE (2)
        "I use it to send my daughter funds and out of town employers send money to me for services rendered.",         # USE (2)
        "Really helped me track my calories intake & be in control of my weight loss journey",                          # USE (2)
        "Helped me track my child.",                                                                                    # USE (2)
        "It had helped me get alot of jobs",                                                                            # USE (2)
        "I hate that my photos get chopped off when uploading to instagram.",                                           # DISLIKE_REASON (3)
        "I hate YouTube because it deletes the views of bts",                                                           # DISLIKE_REASON (3)
        "I hate the shuffling feature now.",                                                                            # DISLIKE_REASON (3)
        "I hate it, the quality is beyond bad, the music is trash, and there isn't a diverse selection of music.",      # DISLIKE_REASON (3)
        "the only thing I dislike is the new icon and lay out.",                                                        # DISLIKE_REASON (3)
        "The voice over is wonderful.",                                                                                 # LIKE_REASON (4)
        "I like it's \"background play\" feature because when i play music,i can turn off the screen",                  # LIKE_REASON (4)
        "I love that I'm able to have full access to all emails, even if they're not recent.",                          # LIKE_REASON (4)
        "It's a cool app and I love the comedy!",                                                                       # LIKE_REASON (4)
        "I love the different stories local, national & international",                                                 # LIKE_REASON (4)
        "I recommend for all of you to think play this game.",                                                          # SIMPLE_RECOMMENDATION (5)
        "I recommend it 100%, completely useful",                                                                       # SIMPLE_RECOMMENDATION (5)
        "Totally recommend getting Netflix ,",                                                                          # SIMPLE_RECOMMENDATION (5)
        "I Would Recommend",                                                                                            # SIMPLE_RECOMMENDATION (5)
        "10/10 definitely recommend",                                                                                   # SIMPLE_RECOMMENDATION (5)
        "Super fun,",                                                                                                   # SIMPLE_LIKE (6)
        "Super app",                                                                                                    # SIMPLE_LIKE (6)
        "I love it so so much",                                                                                         # SIMPLE_LIKE (6)
        "I love it",                                                                                                    # SIMPLE_LIKE (6)
        "I love the app",                                                                                               # SIMPLE_LIKE (6)
        "Worst site",                                                                                                   # SIMPLE_DISLIKE (7)
        "Worst update!!",                                                                                               # SIMPLE_DISLIKE (7)
        "I hate it",                                                                                                    # SIMPLE_DISLIKE (7)
        "so i hate this game....",                                                                                      # SIMPLE_DISLIKE (7)
        "I hate this app..",                                                                                            # SIMPLE_DISLIKE (7)
    ]
    y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]

    # # Remove the manually provided examples from the unseen data
    # for sample in X_train:
    #     if sample in unlabeled_df['text'].values:
    #         unlabeled_df = unlabeled_df[unlabeled_df['text'] != sample]

    for sample in X_train:
        cache.append(sample)

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Check if resuming training and load the saved classifier if needed
if resume_train and os.path.exists(model_path):
    classifier = joblib.load(model_path)
    print("Loaded existing classifier for further training.")
else:
    # Generate embeddings for training data
    X_train_embeddings = np.array(model.encode(X_train, show_progress_bar=True)).reshape(-1, 384)

    # Initialize and train the classifier
    classifier = LogisticRegression(max_iter=1000, verbose=2)
    if len(X_train_embeddings.shape) == 1 or len(X_train_embeddings) == 0:
        raise ValueError("No training data available or training data is incorrectly shaped.")
    if len(X_train) > 0:
        classifier.fit(X_train_embeddings, y_train)

# Initialize empty lists for labeled data
X_train_full = X_train if not resume_train else list(seen_df["text"].to_list())
y_train_full = y_train if not resume_train else list(seen_df['label'].to_list())

# Interactive active learning loop
batch_size = 4
confidence_threshold = 0.6

# Generate embeddings for the unlabeled data
unlabeled_texts = unlabeled_df['text'].tolist()
unlabeled_embeddings = model.encode(unlabeled_texts, show_progress_bar=True) if not unlabeled_df.empty else None

while True:
    # If there are no more unlabeled samples, break the loop
    if unlabeled_df.empty:
        print("No more unlabeled samples available for training.")
        break

    # Generate updated embeddings and text list for the remaining unlabeled data
    unlabeled_texts = unlabeled_df['text'].tolist()
    for item in cache:
        unlabeled_texts.remove(item)
    unlabeled_embeddings = model.encode(unlabeled_texts, show_progress_bar=True) if not unlabeled_df.empty else None

    # Predict probabilities for the unlabeled data
    y_pred_proba = classifier.predict_proba(unlabeled_embeddings)

    # Select uncertain samples based on prediction probabilities
    uncertain_samples = []
    count_certain = 0
    for i, probabilities in enumerate(y_pred_proba):
        max_prob = max(probabilities)
        second_max_prob = sorted(probabilities)[-2]

        # Calculate the uncertainty measure (difference between top two probabilities)
        if max_prob - second_max_prob < confidence_threshold:
            uncertain_samples.append((i, unlabeled_texts[i], max_prob, probabilities))
        else:
            count_certain += 1

    print("Confident in: ", count_certain)

    # If there are no uncertain samples left, break the loop
    if len(uncertain_samples) == 0:
        print("No more uncertain samples below the confidence threshold.")
        break

    # Ask the user to select which label to focus on
    label_choice = input(f"Would you like to focus on a specific label? (0-7) or 'no': \n"
                         "0 - BUG\n"
                         "1 - FEATURE\n"
                         "2 - LIKE_REASON\n"
                         "3 - DISLIKE_REASON\n"
                         "4 - SIMPLE_RECOMMENDATION\n"
                         "5 - SIMPLE_LIKE\n"
                         "6 - SIMPLE_DISLIKE\n")

    if label_choice.lower() == 'no':
        selected_label = None
    else:
        selected_label = int(label_choice)

    # Filter uncertain samples based on the selected label if chosen
    if selected_label is not None:
        uncertain_samples = [
            (i, sample, confidence, probs) for i, sample, confidence, probs in uncertain_samples
            if np.argmax(probs) == selected_label
        ]
        if len(uncertain_samples) == 0:
            print(f"No uncertain samples found for label {selected_label}, showing all uncertain samples instead.")
            uncertain_samples = [
                (i, sample, confidence, probs) for i, sample, confidence, probs in uncertain_samples
            ]

    # Select a batch of uncertain samples for manual labeling
    batch = uncertain_samples[:batch_size]
    for idx, sample, confidence, probs in batch:
        predicted_label = np.argmax(probs)  # Get the predicted label
        print(f"\nSentence: \"{sample}\"")
        print(f"Model is uncertain (confidence: {confidence:.2f})")
        print(f"Probabilities: {probs}")
        print(f"Predicted label: {predicted_label}")

        # Simulate manual labeling process
        label = input(f"""
                      Enter the correct label for this sample (0-7)
                      # 0 - BUG
                      # 1 - FEATURE
                      # 2 - LIKE_REASON
                      # 3 - DISLIKE_REASON
                      # 4 - SIMPLE_RECOMMENDATION
                      # 5 - SIMPLE_LIKE
                      # 6 - SIMPLE_DISLIKE
                      Predicted: {predicted_label}
                      -1 - unsure | skip

                      Input: """)

        if (label in ["0", "1", "2", "3", "4", "5", "6"]):
            # Add labeled sample to the full dataset
            X_train_full.append(sample)
            y_train_full.append(int(label))

        # Remove the labeled sample from the unlabeled dataframe
        cache.append(sample)

    # After updating the labels, retrain the model
    # Perform a stratified split to ensure all classes are represented in both train and test sets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X_train_full, y_train_full):
        X_train = [X_train_full[i] for i in train_idx]
        y_train = [y_train_full[i] for i in train_idx]
        X_test = [X_train_full[i] for i in test_idx]
        y_test = [y_train_full[i] for i in test_idx]

    # Update embeddings for the training data
    X_train_embeddings = np.array(model.encode(X_train, show_progress_bar=True)).reshape(-1, 384)

    # Retrain the classifier with the updated training dataset
    classifier.fit(X_train_embeddings, y_train)

    # After retraining, show the metrics once
    X_test_embeddings = np.array(model.encode(X_test, show_progress_bar=True)).reshape(-1, 384)

    if len(X_test_embeddings) > 0:
        y_pred = classifier.predict(X_test_embeddings)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nUpdated Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=[
            "BUG", "FEATURE", "LIKE_REASON", "DISLIKE_REASON", "SIMPLE_RECOMMENDATION", "SIMPLE_LIKE", "SIMPLE_DISLIKE"
        ]))

        # for text, label_test, label_pred in zip(X_test, y_test, y_pred):
        #     print(f"{text}: test ({label_test}) | ({label_pred})")

        result_string = f"""
\nUpdated Accuracy: {accuracy * 100:.2f}%\n
\nClassification Report:\n", {classification_report(y_test, y_pred, target_names=[
            "BUG", "FEATURE", "LIKE_REASON", "DISLIKE_REASON", "SIMPLE_RECOMMENDATION", "SIMPLE_LIKE", "SIMPLE_DISLIKE"
        ])}\n
"""
    else:
        print("Test embeddings are empty, skipping evaluation.")

    con = input("Continue model? (y|n): ")

    if con == "n":
        break

# Save the retrained classifier
joblib.dump(classifier, model_path)

# Save all remaining unlabeled data to unseen.txt
if (len(cache) != 0):
    with open(unseen_path, 'w', encoding='utf-8') as file:
        for text in unlabeled_df['text'].tolist():
            if (text not in cache):
                file.write(f"{text}\n")

# Save all labeled data to seen.csv
df = pd.DataFrame({
    'text': X_train_full,
    'label': y_train_full
})

# Sort by 'label' column
df = df.sort_values(by='label')

df.to_csv(seen_path, index=False)

with open("report_classifier.txt", "w", encoding="utf-8") as f:
    f.write(result_string)