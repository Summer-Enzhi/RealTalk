from argparse import ArgumentParser
import json
from src.evaluator import evaluator_mapping
from loguru import logger
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--evaluator', type=str, required=True, choices=list(evaluator_mapping.keys()))
    args = parser.parse_args()
    data = []
    with open(args.src_file, 'r') as f:
        # for line in f:
        #     json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
        #     data.append(json_obj)
        data = json.load(f)
    evaluator = evaluator_mapping[args.evaluator]()
    res = evaluator.evaluate(data)
    
    dir = "/".join(args.src_file.split("/")[:-1])
    file = args.src_file.split("/")[-1].replace(".json","_metrics.json")
    with open(f'{dir}/{file}', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    


if __name__ == "__main__":
    main()
