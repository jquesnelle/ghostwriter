import argparse
import json
import sys
import random
import os
import torch
import ftfy
import re
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib import tenumerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.backends.cuda.matmul.allow_tf32 = True


class StorySummarizer():
    _SUMMARY_PROMPT = "\n\n### SUMMARY:\n"
    _ANALYSIS_PROMPT = "\n\n### ANALYSIS:\n"

    def __init__(self, model_name_or_path, penalty_alpha, top_k, max_new_tokens, load_in_4bit):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            quantization_config=nf4_config if load_in_4bit else None, device_map="auto"
        )

        self._inference_params = {
            "penalty_alpha": penalty_alpha,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def set_inference_params(self, params):
        if "max_length" in params:
            params["max_new_tokens"] = params["max_length"]
            del params["max_length"]
        self._inference_params.update(params)

    def summarize(self, input_text):
        return self._run(input_text, self._SUMMARY_PROMPT)

    def analyze(self, input_text):
        return self._run(input_text, self._ANALYSIS_PROMPT)

    def _run(self, input_text, prompt):
        input_text += prompt
        input_ids = self.tokenizer(
            input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, **self._inference_params)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        loc = answer.find(prompt)
        answer = answer[loc + len(prompt):]
        return answer

# from https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/utils/text.py


def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv


def main(args):
    dataset = load_dataset(args.dataset)["train"]

    ids = dataset["id"]

    if args.chunks > 1:
        if args.validation_split != 0 or args.test_split != 0:
            print("Note, validation/test splits will not be chunked")

    index = {id: index for (index, id) in enumerate(ids)}

    if args.shard is not None:
        parts = args.shard.split(",")
        this_shard = int(parts[0])
        num_shards = int(parts[1])
        assert args.chunks == 1
        assert num_shards > 1
        assert this_shard >= 0 and this_shard < num_shards

        ids = [ids[i]
               for i in range(0, len(ids)) if (i % num_shards) == this_shard]

    tokenizer = AutoTokenizer.from_pretrained(
        args.stats_for_tokenizer) if args.stats_for_tokenizer is not None else None
    tokenizer_stats = {
        "num_text_tokens": [],
        "num_summary_tokens": [],
        "num_text_and_summary_tokens": []
    }

    summarizer = StorySummarizer(
        args.summarizer, args.penalty_alpha, args.top_k, args.summary_max_tokens, args.load_in_4bit) if not args.no_summary else None

    data = []
    for id in tqdm(ids, desc="Books"):

        i = index[id]
        entry = dataset[i]
        title: str = entry["title"]
        author: str = entry["author"]
        utf8: str = entry["utf8"]

        if not args.no_ftfy:
            utf8 = ftfy.fix_text(utf8)

        passages = split_and_recombine_text(
            utf8, args.passage_desired_len, args.passage_max_len)

        for i, passage in tenumerate(passages, desc=f"Passages (Title=\"{title}\", Author=\"{author}\")", leave=False):

            summary = summarizer.summarize(passage).strip().replace("\n", "") if not args.no_summary else ""

            if tokenizer is not None:
                text_tokens = len(tokenizer.encode(
                    passage, add_special_tokens=False))
                summary_tokens = len(tokenizer.encode(
                    summary, add_special_tokens=False))
                tokenizer_stats["num_text_tokens"].append(text_tokens)
                tokenizer_stats["num_summary_tokens"].append(summary_tokens)
                tokenizer_stats["num_text_and_summary_tokens"].append(
                    text_tokens + summary_tokens)
                
            data.append(json.dumps({
                "passage": passage,
                "summary": summary,
                "title": title,
                "author": author,
                "id": id,
                "passage_index": i
            }))

    validation_size = int(len(data) * args.validation_split)
    test_size = int(len(data) * args.test_split)

    validation = data[0:validation_size]
    test = data[validation_size: test_size + validation_size]
    train = data[test_size + validation_size:]

    print(f"# Train: {len(train)}")
    if validation_size > 0:
        print(f"# Valid: {validation_size}")
    if test_size > 0:
        print(f"# Valid: {test_size}")

    output_name = f"{args.dataset.split('/')[1]}-summarized"

    output_dir = os.path.join(args.output_dir, output_name, "data")
    os.makedirs(output_dir, exist_ok=True)

    shard_suffix = f"-{this_shard+1:05}" if args.shard is not None else ""
    if args.chunks <= 1:
        open(os.path.join(output_dir, f"train{shard_suffix}.jsonl"),
             "w", encoding="utf-8").writelines(train)
    else:
        chunk_size = int(len(train) / args.chunks)
        if (len(train) % args.chunks) != 0:
            chunk_size += 1
        train = [train[i:i + chunk_size]
                 for i in range(0, len(train), chunk_size)]
        for i in range(0, len(train)):
            open(os.path.join(output_dir, f"train-{i+1:05}.jsonl"),
                 "w", encoding="utf-8").writelines(train[i])
    if len(validation) > 0:
        open(os.path.join(output_dir, f"valid{shard_suffix}.jsonl"),
             "w", encoding="utf-8").writelines(validation)
    if len(test) > 0:
        open(os.path.join(output_dir, f"test{shard_suffix}.jsonl"),
             "w", encoding="utf-8").writelines(test)

    if tokenizer is not None:
        import numpy
        for key, value in tokenizer_stats.items():
            print(f"{key}: min: {numpy.min(value)} max: {numpy.max(value)} avg: {int(numpy.average(value))} med: {int(numpy.median(value))} total: {numpy.sum(value)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)

    parser.add_argument("--stats_for_tokenizer", type=str)
    parser.add_argument("--stats_only", type=str)

    parser.add_argument("--summarizer", type=str,
                        default="emozilla/mpt-7b-storysummarizer")
    parser.add_argument("--penalty_alpha", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--summary_max_tokens", type=int, default=128)
    parser.add_argument("--no_summary", action="store_true")

    parser.add_argument("--no_ftfy", action="store_true")
    parser.add_argument("--passage_desired_len", type=int, default=1500)
    parser.add_argument("--passage_max_len", type=int, default=2000)

    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--validation_split", type=float, default=0)
    parser.add_argument("--test_split", type=float, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--shard", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
