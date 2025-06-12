from simalign import SentenceAligner
from collections import Counter
import Levenshtein
import pandas as pd
import random
import spacy
import json

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

def tokenize(text: str, lang: str) -> list[str]:
    nlp = nlp_en if lang == "en" else nlp_fr
    return text.split()
    # return [token.lemma_ for token in nlp(text) if not token.is_punct]
    # return [token.text for token in nlp(text) if not token.is_space and not token.is_punct]

def load_aligned_sentences(filepath: str) -> list[tuple[str, str]]:
    df = pd.read_csv(filepath, sep="\t", encoding="utf-8").fillna("")
    return df.values.tolist()

def find_subsequence_indices(tokens: list[str], subseq: list[str]) -> list[int]:
    return [i for i in range(len(tokens) - len(subseq) + 1) if tokens[i:i+len(subseq)] == subseq]

def split_continuous_indices(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    indices = sorted(set(indices))
    groups = [[indices[0]]]
    for idx in indices[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups

def extract_french_name_candidates(en_sent: str, fr_sent: str, en_name: str, alignments: dict) -> list[str]:
    en_tokens = tokenize(en_sent, "en")
    fr_tokens = tokenize(fr_sent, "fr")
    name_tokens = tokenize(en_name, "en")
    name_start_indices = find_subsequence_indices(en_tokens, name_tokens)

    matched = alignments["mwmf"] + alignments["inter"] + alignments["itermax"]
    candidates = []

    for start in name_start_indices:
        en_indices = range(start, start + len(name_tokens))
        fr_indices = [fr_idx for en_idx, fr_idx in matched if en_idx in en_indices]
        if not fr_indices:
            continue
        for group in split_continuous_indices(fr_indices):
            candidates.append(" ".join(fr_tokens[i] for i in group))

    return candidates

def find_french_equivalents(
        name: str, sentence_pairs: list[tuple[str, str]], aligner, n_samples: int = 10
) -> tuple[str | None, dict[str, int]]:
    relevant_pairs = [(en, fr) for en, fr in sentence_pairs if name in en]
    sampled_pairs = random.sample(relevant_pairs, min(n_samples, len(relevant_pairs)))

    all_candidates = []
    for en_sent, fr_sent in sampled_pairs:
        alignments = aligner.get_word_aligns(en_sent, fr_sent)
        candidates = extract_french_name_candidates(en_sent, fr_sent, name, alignments)
        all_candidates.extend(candidates)

    counter = Counter(all_candidates)
    if not counter:
        return None, {}

    most_common = counter.most_common()
    top_count = most_common[0][1]
    top_candidates = [cand for cand, count in most_common if count == top_count]

    if len(top_candidates) == 1:
        best_match = top_candidates[0]
    else:
        # Tie-breaking: pick closest to the original name
        best_match = min(top_candidates, key=lambda cand: Levenshtein.distance(name, cand))

    return best_match, dict(counter)

def save_name_mapping(mapping: dict[str, str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def main():
    aligned_ds_file = "emma_pred.csv"
    characters_file = "emma_char_info.csv"
    output_json = "emma_map.json"

    character_names = pd.read_csv(characters_file)["Main Name"].tolist()
    n_samples = 100

    sentence_pairs = load_aligned_sentences(aligned_ds_file)
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    name_mapping = {}

    for name in character_names:
        best_match, candidates_dict = find_french_equivalents(name, sentence_pairs, aligner, n_samples)
        count = candidates_dict.get(best_match, 0)
        if best_match:
            name_mapping[name] = best_match
        print(f"\n🔎 English: '{name}' → 🇫🇷 French: **{best_match}** (count: {count})")
        print(f"All candidates: {candidates_dict}")

    save_name_mapping(name_mapping, output_json)
    print(f"\n✅ Name mapping saved to {output_json}")

if __name__ == "__main__":
    main()
