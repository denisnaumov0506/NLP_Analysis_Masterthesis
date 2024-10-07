from transformers import RobertaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, RobertaForTokenClassification
import torch
import pandas as pd
import torch.nn.functional as F

# load data

print('Loading data...')

data = []

app = "twitch"

with open(f"H:\Laptop\last_3yeras_50k\{app}_results.txt", "r", encoding='utf-8') as f:
# with open(f"SequenceClassification\\text_classification_final.txt", "r", encoding='utf-8') as f:
    for line in f.readlines():
        data.append(line.strip())

# data = data[0:100_000]

print("Data loaded")

# Split the reviews into sentences!

# Load the model and tokenizer
print("Loading sentence splitter model...")
model = RobertaForTokenClassification.from_pretrained('./TokenClassification/saved_model_7967')
model.to('cuda')
tokenizer = RobertaTokenizer.from_pretrained('./TokenClassification/saved_model_7967')
print("Sentence splitter loaded")

# Mapping of IDs to labels
id2label = {
    0: "SBW",
    1: "TPM",
    2: "SEW",
    3: "CW",
    4: "NTPM"
}

# Set batch size
batch_size = 60
sentences_all = []

# Process data in batches
for start_idx in range(0, len(data), batch_size):
    end_idx = min(start_idx + batch_size, len(data))
    batch_data = data[start_idx:end_idx]

    # Tokenize the inputs
    tokenized_inputs = tokenizer(batch_data, is_split_into_words=False, return_tensors="pt", truncation=True, padding=True)
    tokenized_inputs = {key: value.to('cuda') for key, value in tokenized_inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs.logits

    # Get the predicted labels
    predictions = torch.argmax(logits, dim=2)

    # Split sentences in the batch
    for i in range(len(batch_data)):
        # Convert input IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i].tolist())

        # Convert predictions to labels
        pred_labels = [id2label[p.item()] for p in predictions[i]]
        
        pred_labels = [label for label, token in zip(pred_labels, tokens) if token not in ['<s>', '</s>', '<pad>']]
        tokens = [token for token in tokens if token not in ['<s>', '</s>', '<pad>']]

        tokens = [token.replace("Ġ", " ") for token in tokens]
        pred_labels = [label for label in pred_labels]

        sentence_boundaries = []

        for index, label in enumerate(pred_labels):
            if (label == 'SEW' and (index == len(pred_labels) - 1 or pred_labels[index + 1] != 'TPM')):
                sentence_boundaries.append((index, label))
            elif (label == "TPM"):
                sentence_boundaries.append((index, label))

        last = 0

        local_sents = []

        for idx, boundary in enumerate(sentence_boundaries):
            sentence = "".join(tokens[last:boundary[0] + 1])
            last = boundary[0] + 1
            local_sents.append(sentence.strip())

        sentences_all.append(local_sents)

    print(f"Processed {end_idx}/{len(data)} reviews")

print("All reviews processed")

print("Saving reviews")

# with open(f"H:\Laptop\last_3yeras_50k\{app}_results_splitted.txt", "w", encoding='utf-8') as f:
with open(f"SequenceClassification\\text_classification_final_splitted.txt", "w", encoding='utf-8') as f:
    for item in sentences_all:
        for sent in item:
            f.write(sent)
            f.write("\n")
        # f.write("\n")