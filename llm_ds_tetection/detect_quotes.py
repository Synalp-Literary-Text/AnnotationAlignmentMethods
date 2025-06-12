import os
import time
from typing import List

import ollama
import pandas as pd
from pydantic import BaseModel, RootModel

model = "hf.co/MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF:Q8_0"
ollama.pull(model)
print("Model is ready")

PROMPT_TEMPLATE = """
Tu es un assistant expert en littérature française et en annotation structurée de discours direct.

Ta tâche est d’extraire toutes les instances de discours direct (dialogue, monologue interne, exclamations citées) contenues dans un paragraphe littéraire français.

Pour chaque citation directe identifiée, retourne un objet JSON contenant les champs suivants :

- "quote_text" : le texte complet de la citation, même si elle est interrompue par de la narration
- "sub_quotations_list" : liste des segments ininterrompus du discours
- "referring_expression" : l’expression narrative qui fait référence au locuteur (par exemple : "dit-il", "s’écria-t-elle") ; si elle est absente, utilise ""

Seulement le discours direct est permis, c’est-à-dire :
- le texte entre guillemets français (« »), guillemets anglais (“ ”), ou précédé de tirets cadratins (—)
- les citations doivent exclure toute narration ou discours indirect

Si le paragraphe ne contient aucun discours direct, retourne :

```json
{{ "root": [] }}
````

Le format de sortie est toujours du JSON **valide** structuré comme ceci :

```json
{{
  "root": [
    {{
      "quote_text": "...",
      "sub_quotations_list": ["...", "..."],
      "referring_expression": "..."
    }}
  ]
}}
```

### **Exemples illustratifs des cinq cas de discours direct**

---

// **Exemple 1 — Discours avec guillemets + incise complexe + narration intercalée**
Texte :
« Je t’assure que tout ira bien. Ce n’est qu’une question de temps — dit-il en souriant, posant une main sur son épaule — et je resterai à tes côtés. » Elle baissa les yeux, sans dire un mot.

```json
{{
  "root": [
    {{
      "quote_text": "Je t’assure que tout ira bien. Ce n’est qu’une question de temps et je resterai à tes côtés.",
      "sub_quotations_list": [
        "Je t’assure que tout ira bien.",
        "Ce n’est qu’une question de temps",
        "et je resterai à tes côtés."
      ],
      "referring_expression": "dit-il en souriant, posant une main sur son épaule"
    }}
  ]
}}
```

---

// **Exemple 2 — Dialogue entre deux personnages avec incises et narration**
Texte :
« Tu comptes vraiment partir demain ? demanda-t-elle d’une voix tremblante. Je pensais que tu attendrais au moins la fin du mois. » Il détourna les yeux, prit une longue inspiration et répondit : « Je n’ai plus le choix. Tout est prêt. » Le silence s’installa, pesant.

```json
{{
  "root": [
    {{
      "quote_text": "Tu comptes vraiment partir demain ? Je pensais que tu attendrais au moins la fin du mois.",
      "sub_quotations_list": [
        "Tu comptes vraiment partir demain ?",
        "Je pensais que tu attendrais au moins la fin du mois."
      ],
      "referring_expression": "demanda-t-elle d’une voix tremblante"
    }},
    {{
      "quote_text": "Je n’ai plus le choix. Tout est prêt.",
      "sub_quotations_list": [
        "Je n’ai plus le choix.",
        "Tout est prêt."
      ],
      "referring_expression": "répondit"
    }}
  ]
}}
```

---

// **Exemple 3 — Dialogue avec tiret cadratin et incise narrative**
Texte :
— Je ne reviendrai pas, murmura-t-elle, les yeux pleins de larmes. Ce que j’ai vu là-bas m’a changée à jamais. Je ne peux pas faire semblant.

```json
{{
  "root": [
    {{
      "quote_text": "Je ne reviendrai pas. Ce que j’ai vu là-bas m’a changée à jamais. Je ne peux pas faire semblant.",
      "sub_quotations_list": [
        "Je ne reviendrai pas.",
        "Ce que j’ai vu là-bas m’a changée à jamais.",
        "Je ne peux pas faire semblant."
      ],
      "referring_expression": "murmura-t-elle"
    }}
  ]
}}
```

---

// **Exemple 4 — Narration continue, aucun discours direct**
Texte :
Le vent soufflait fort sur la lande déserte, soulevant des tourbillons de poussière sèche. Les nuages s’amoncelaient à l’horizon, lourds de promesses de tempête. Personne ne semblait en vue, et le silence paraissait peser sur le paysage.

```json
{{
  "root": []
}}
```

---

// **Exemple 5 — Monologue continu d’un seul personnage**
Texte :
« Je t’attendais. Pourquoi as-tu tardé ? Tu savais pourtant que c’était important. Maintenant tout est compromis, et je ne sais pas comment réparer ça. »

```json
{{
  "root": [
    {{
      "quote_text": "Je t’attendais. Pourquoi as-tu tardé ? Tu savais pourtant que c’était important. Maintenant tout est compromis, et je ne sais pas comment réparer ça.",
      "sub_quotations_list": [
        "Je t’attendais.",
        "Pourquoi as-tu tardé ?",
        "Tu savais pourtant que c’était important.",
        "Maintenant tout est compromis, et je ne sais pas comment réparer ça."
      ],
      "referring_expression": ""
    }}
  ]
}}
```

---

Maintenant, traite un paragraphe littéraire en entrée et retourne uniquement le résultat JSON dans le format ci-dessus:
```input
{text}
```
"""


class Quote(BaseModel):
    quote_text: str
    sub_quotations_list: List[str]
    referring_expression: str

class QuotesList(RootModel[List[Quote]]):
    """Liste des partitions"""


def extract_quotes(paragraph: str) -> List[Quote]:
    prompt = PROMPT_TEMPLATE.format(text=paragraph)

    start = time.perf_counter()
    response = ollama.generate(
        model=model,
        prompt=prompt,
        format=QuotesList.model_json_schema(),
        options={
            "temperature": 0,
            "lang": "fr",
            "seed": 42,
        },
        stream=False
    ).response
    print(f"LLM response took: {time.perf_counter() - start:.2f}s")
    return QuotesList.model_validate_json(response).root


def process_file(filename: str):
    with open(os.path.join("chapters", filename), "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = content.split("\n\n")

    quotes = []
    for paragraph in paragraphs:
        extracted = extract_quotes(paragraph)
        for quote in extracted:
            if quote.quote_text:
                quotes.append(quote)

    quotes = [
        {
            "quoteID": f"Q{i}",
            "quoteText": quote.quote_text,
            "subQuotationList": quote.sub_quotations_list,
            "referringExpression": quote.referring_expression
        }
        for i, quote in enumerate(quotes)
    ]

    df = pd.DataFrame(quotes)
    df.to_csv(os.path.join("quotes", filename.replace(".txt", ".csv")), sep="\t", index=False)

files = sorted(os.listdir("chapters"))
for file in files:
    if file.endswith(".txt"):
        start = time.perf_counter()
        process_file(file)
        print(f"Processed {file} in {time.perf_counter() - start:.2f}s")
