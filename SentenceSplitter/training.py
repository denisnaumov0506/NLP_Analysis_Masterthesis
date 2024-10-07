from datasets import Dataset
import json
from transformers import RobertaForTokenClassification, RobertaTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# Load the JSON dataset
with open('TokenClassification/labeled_dataset.json', 'r') as f:
    data = json.load(f)

# Convert the list of dictionaries to a dictionary of lists
def list_of_dicts_to_dict_of_lists(data):
    columns = data[0].keys()
    dict_of_lists = {column: [entry[column] for entry in data] for column in columns}
    return dict_of_lists

# Convert the dataset
data_dict = list_of_dicts_to_dict_of_lists(data)
dataset = Dataset.from_dict(data_dict)

# Split the dataset into training and evaluation sets
train_test_split = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Tokenize the dataset and align labels with tokens
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples['text'], 
        is_split_into_words=False, 
        padding='max_length', 
        truncation=True, 
        max_length=512  # Set max_length to 512
    )
    
    padded_labels = [
        [-100] + label + [-100] + [-100] * (512 - len(label) - 2) for label in examples['labels']
    ]
    
    tokenized_inputs["labels"] = padded_labels
    tokenized_inputs["text"] = examples["text"]
    tokenized_inputs["tokens"] = examples["tokens"]
    return tokenized_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

id2label = {
    0: "SBW", # sentence beginning word
    1: "TPM", # terminating punctuation mark
    2: "SEW", # sentence ending word
    3: "CW", # common word
    4: "NTPM" # non-terminating punctuation mark
}
label2id = {
    "SBW": 0,
    "TPM": 1,
    "SEW": 2,
    "CW": 3,
    "NTPM": 4
}

# Load model
model = RobertaForTokenClassification.from_pretrained('roberta-large', num_labels=5, id2label=id2label, label2id=label2id)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./TokenClassification/results',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,  # Log every 100 steps
    load_best_model_at_end=True
)

label_list = ["SBW", "TPM", "SEW", "CW", "NTPM"]

# # Define metric function
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if (l != -100)]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if (l != -100)]
#         for prediction, label in zip(predictions, labels)
#     ]

#     # print(true_predictions)
#     # print(true_labels)

#     # Compute the precision, recall, F1 score, and accuracy
#     precision = precision_score(true_labels, true_predictions)
#     recall = recall_score(true_labels, true_predictions)
#     f1 = f1_score(true_labels, true_predictions)
#     accuracy = accuracy_score(true_labels, true_predictions)

#     # Print the results
#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "accuracy": accuracy,
#     }

# Define metric function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # print(predictions)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if (l != -100)]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if (l != -100)]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [item for sublist in true_labels for item in sublist]

    true_predictions = [item for sublist in true_predictions for item in sublist]

    # true_predictions = [id2label[p] for sublist in true_predictions for p in sublist]
    # true_labels = [id2label[l] for sublist in labels for l in sublist if l != -100]

    # print(true_predictions)


    # print("true_predictions")
    # print(predictions)
    # print("true_labels")
    # print(labels)

    # print(['FEATURE' if label == pred else 'UNK' for label, pred in zip(true_labels, true_predictions)])

    # Compute the per-class precision, recall, F1 score using classification report
    class_report = classification_report(
        # labels,
        true_labels,  # Flatten lists    
        # predictions,
        true_predictions,
        target_names=label_list,
        output_dict=True
        )
    
    # Extract the per-class precision
    per_class_precision = {label: metrics['precision'] for label, metrics in class_report.items() if label in label_list}

    # Extract the per-class precision
    per_class_recall = {label: metrics['recall'] for label, metrics in class_report.items() if label in label_list}

    # Extract the per-class precision
    per_class_f1 = {label: metrics['f1-score'] for label, metrics in class_report.items() if label in label_list}

    # Compute the weighted average precision, recall, F1 score, and accuracy
    precision_macro = precision_score(true_labels, true_predictions, average='macro')
    recall_macro = recall_score(true_labels, true_predictions, average='macro')
    f1_macro = f1_score(true_labels, true_predictions, average='macro')
    accuracy = accuracy_score(true_labels, true_predictions)

    # Return the results
    return {
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "precision_weighted": precision_macro,
        "recall_weighted": recall_macro,
        "f1_weighted": f1_macro,
        "accuracy": accuracy,
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()

# Get predictions
predictions, labels, _ = trainer.predict(eval_dataset)
predictions = np.argmax(predictions, axis=2)

# Access the text information from the evaluation dataset and remove padding
for i in range(len(eval_dataset)):
    text = eval_dataset[i]['text']
    tokens = eval_dataset[i]['tokens']
    true_labels = eval_dataset[i]['labels']
    pred_labels = predictions[i][1:len(tokens) + 1]  # Remove padding
    true_labels = true_labels[1:len(tokens) + 1]  # Remove padding
    pred_tokens = [id2label[i] for i in pred_labels]

# Save the model and tokenizer
trainer.model.save_pretrained('./TokenClassification/saved_model_7967_duplicate')
tokenizer.save_pretrained('./TokenClassification/saved_model_7967_duplicate')

best_epoch = trainer.state.best_model_checkpoint
best_metric = trainer.state.best_metric
print(f"The best model was from epoch {best_epoch} | {best_metric}")