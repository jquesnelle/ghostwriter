import argparse
import json
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def new_entry():
    return 

def main(args):
    dataset = load_dataset(args.dataset)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data = []
    pending_entry = None
    pending_entry_tokenized = []
    last_passage_index = -1
    
    i = 0
    bar = tqdm(total=len(dataset), desc="Passage")
    while i < len(dataset):
        entry = dataset[i]
        if pending_entry is None:
            if args.sharegpt:
                pending_entry = {"conversations": []}
                pending_entry_tokenized = []
            else:
                pending_entry = {"text": f"{args.header_prompt}"}
                pending_entry_tokenized = tokenizer(pending_entry["text"]).input_ids

        to_add = f"{args.summary_prompt}{entry['summary']}{args.passage_prompt}{entry['passage']}"
        to_add_tokenized = tokenizer(to_add).input_ids
        if args.sharegpt:
            to_add = [{"from": args.summary_prompt, "value": entry["summary"]}, {"from": args.passage_prompt, "value": entry["passage"]}]

        changed_book = entry["passage_index"] != last_passage_index + 1
        if len(to_add_tokenized) + len(pending_entry_tokenized) > args.max_seq_len or changed_book:
            data.append(json.dumps(pending_entry))
            pending_entry = None
            if changed_book:
                last_passage_index = -1
        else:
            if args.sharegpt:
                pending_entry["conversations"].extend(to_add)
            else:
                pending_entry["text"] += to_add
            pending_entry_tokenized.extend(to_add_tokenized)
            last_passage_index = entry["passage_index"]
            i += 1
            bar.update()
        
    print(f"# Train: {len(data)}")

    output_name = f"{args.dataset.split('/')[1]}-{args.tokenizer.split('/')[1]}-{args.max_seq_len}"
    output_dir = os.path.join(args.output_dir, output_name, "data")
    os.makedirs(output_dir, exist_ok=True)
    open(os.path.join(output_dir, f"train.jsonl"),
             "w", encoding="utf-8").writelines(data)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tokenizer", type=str)

    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--summary-prompt", type=str, default="\n\n### Summary:\n")
    parser.add_argument("--passage-prompt", type=str, default="\n\n### Passage:\n")
    parser.add_argument("--header-prompt", type=str, default="Below are subsequent passages from a novel. Continue the story in a creative yet coherent manner according to the preceding summary.")
    parser.add_argument("--sharegpt", action="store_true")

    parser.add_argument("--output_dir", type=str, default="outputs")
    
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
