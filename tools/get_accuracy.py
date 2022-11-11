import argparse
import json

def get_accuracy(truth, target) -> float:
  # compare truth and target and return accuracy
  match = 0
  for idx, question in enumerate(truth):
    if question == target[idx]:
      for candidate in target[idx]['candidates']:
        if candidate == question['answer']:
          match += 1
  return match/len(truth)

def main(args):
  # open json file and read the data into a variable
  truth = None
  target = None
  with open(args.truth) as f:
    truth = json.load(f.read())['data']
  with open(args.target) as f:
    target = json.load(f.read())['data']
  # get accuracy
  accuracy = get_accuracy(truth, target)
  print(accuracy)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--truth', type=str, required=True, default="qatask/database/datasets/wikipedia.jsonl")
  parser.add_argument('--target', type=str, required=True, default="qatask/database/datasets/wikipedia_ans.jsonl")
  args = parser.parse_args()
  main(args)