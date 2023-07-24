import argparse
import sys
from datasets import load_dataset

def main(args):
    ds = load_dataset(args.dataset)
    ds = ds.map(lambda x: {"text": f"{x['chapter']}\n\n### {x['type'].upper()}:\n{x['text']}"}, remove_columns=["chapter", "type"])
    print(ds)

    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub, private=args.private)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--push-to-hub", type=str)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
