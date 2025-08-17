from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import pandas as pd
from datasets import load_dataset
from collections import defaultdict


def clean_text_preserve_case(text):
    if pd.isna(text):
        return ""

    #remove quotes, brackets, curly braces, parentheses
    text_clean = re.sub(r'["\'\[\]\{\}\(\)]', '', text)

    #replace separators (&, commas) with slash
    text_clean = re.sub(r'[,&]+', '/', text_clean)

    #collapse multiple slashes
    text_clean = re.sub(r'/+', '/', text_clean)

    #remove spaces around slashes
    text_clean = re.sub(r'\s*/\s*', '/', text_clean)

    #strip leading/trailing slashes and whitespace
    text_clean = text_clean.strip(' /')

    return text_clean



def batch_extract_person_names(nlp, texts):
    #clean all texts first
    cleaned_texts = [clean_text_preserve_case(t) for t in texts]

    #split all cleaned texts into chunks and keep track of which chunks belong to which text
    all_chunks = []
    idx_map = []  # mapping data structure from chunk index to original text index
    for i, text in enumerate(cleaned_texts):
        chunks = text.split('/') if text else []
        all_chunks.extend(chunks)
        idx_map.extend([i] * len(chunks))

    #run NER on all chunks at once
    ner_results = nlp(all_chunks)

    #extract person name
    chunk_person_names = []
    for entities in ner_results:
        person_parts = [ent["word"].strip() for ent in entities if ent["entity_group"] == "PER"]
        chunk_person_names.append(" ".join(person_parts) if person_parts else "")

    #map back to original text
    result = [""] * len(texts)
    agg = defaultdict(list)
    for idx, name in zip(idx_map, chunk_person_names):
        if name:
            agg[idx].append(name)

    for idx, names in agg.items():
        result[idx] = "/".join(names)
    return result

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy = "simple", device = 'cuda:0')

    col_name = "raw_comp_writers_text"
    dataset = load_dataset("csv", data_files="normalization_assesment_dataset_10k.csv", split="train")
    dataset = dataset.select_columns([col_name])

    dataset = dataset.map(
        lambda batch: {"CLEAN_TEXT": batch_extract_person_names(nlp, batch[col_name])},
        batched=True,
        batch_size=512
    )
    dataset.to_csv("clean_comp_writers_text.csv", index=False)

