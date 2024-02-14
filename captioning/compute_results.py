from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score_compute
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--predict-json", required=True, help="prediction json file.")
    parser.add_argument("--target-json", required=True, help="ground truth json file.")
    args = parser.parse_args()
    
    pd = json.load(open(args.predict_json))
    gt = json.load(open(args.target_json))
    pds = defaultdict(list)
    gts = defaultdict(list)
    pds_all = []
    gts_all = []

    for i, data in enumerate(gt):
        video, captions = data["video"], data["caption"]
        pds[i].append({"image_id":video, "caption":pd[video]})
        pds_all += ([pd[video]]*len(captions))

        for caption in captions:
            gts[i].append({"image_id":video, "caption":caption})
        gts_all += captions

    tokenizer = PTBTokenizer()
    pds = tokenizer.tokenize(pds)
    gts = tokenizer.tokenize(gts)
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
               (Meteor(),"METEOR"),
               (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]

    eval_dict = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, pds)
        if scorer.method() == "Bleu":
            eval_dict["BLEU4"] = score[3]
        else:
            eval_dict[scorer.method()] = score

    _, _, score = bert_score_compute(pds_all, gts_all, lang='en', verbose=False)
    eval_dict["BERTScore"] = score.mean().item()

    for k, v in eval_dict.items():
        print("%s: %.2f%%"%(k, v*100))
