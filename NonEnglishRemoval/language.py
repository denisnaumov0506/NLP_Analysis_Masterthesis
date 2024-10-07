import pycld2 as cld2
import pandas as pd
from cleantext import clean

app = "twitch"

df = pd.read_csv(f"/mnt/c/Users/Denis/repos/PythonScripts/last_3yeras_50k/{app}_first_50k_reviews_last_3_years.csv")

texts = list(df['text'].to_list())

count = 0

english_texts = []
other_texts = []

for text in texts:
    count += 1

    if (count % 100 == 0):
        print(f"Processed: {count}/{len(texts)}")

    text = clean(text, no_emoji=True)

    if (len(text) == 0):
        continue

    isReliable, textBytesFound, details = cld2.detect(
        text
    )

    # print(isReliable)

    # print(details[0][1] == "en")

    isEnglish = False

    lang = details[0][1]

    if (lang == "en"):
        isEnglish = True

    if isEnglish == True:
         english_texts.append(text)
        
    elif isEnglish == False:
        if (len(text.split()) > 6):
            other_texts.append(text)
        else:
            english_texts.append(text)
        

with open(f"/mnt/c/Users/Denis/repos/PythonScripts/last_3yeras_50k/{app}_results.txt", "w", encoding="utf-8") as f:
    for item in english_texts:
        f.write(item)
        f.write("\n")

with open(f"/mnt/c/Users/Denis/repos/PythonScripts/last_3yeras_50k/{app}_results_other.txt", "w", encoding="utf-8") as f:
    for item in other_texts:
        f.write(item)
        f.write("\n")