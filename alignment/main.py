import os
import pandas as pd

def en(fname):
    df = pd.read_csv("data/quote_annotations/fr/" + fname, encoding="utf-8", sep="\t")
    df["quoteText"] = df["quoteText"].apply(lambda x: x.replace("\n", " ").replace("\r", ""))
    with open("data/quotes/fr/" + fname[:-4] + ".txt", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(df["quoteText"].values.tolist())
        )

for i in range(1, 6):
    en(f"Emma_chap{i}.csv")
