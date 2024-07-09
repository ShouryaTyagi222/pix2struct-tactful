import os
import sys
sys.path.append("../")
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
import Levenshtein

import sys
sys.path.append("../")

from configs import *


def cal_scores(outputs, labels, documents, processor_base):
    total_rouge1=0
    total_rougeL=0
    total_anls=0
    total_samples = 0

    # Decode predictions and labels
    predictions = processor_base.batch_decode(outputs.logits.argmax(-1))
    labels = processor_base.batch_decode(labels)

    # Calculate ROUGE scores
    for pred, label in zip(predictions, labels):
        eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
        eos_label = label.index("</s>") if "</s>" in label else len(label)

        pred_no_pad = [token for token in pred[:eos_pred] if token != '<pad>']
        label_no_pad = [token for token in label[:eos_label] if token != '<pad>']

        scores = scorer.score(" ".join(pred_no_pad), " ".join(label_no_pad))
        total_rouge1 += scores['rouge1'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        total_samples += 1

        dist = Levenshtein.distance(" ".join(pred_no_pad), " ".join(label_no_pad))
        normalized_dist = dist / len(label[:eos_label])
        total_anls = total_anls + normalized_dist
    
    # Write predictions and labels to file
    with open(os.path.join(model_path,f"{model_name}_sample_q_and_a.txt"), 'a') as f:
        for doc, pred, label in zip(documents, predictions, labels):
            eos_pred = pred.index("</s>") if "</s>" in pred else len(pred)
            eos_label = label.index("</s>") if "</s>" in label else len(label)

            f.write("Document: " + str(doc) + "\n")
            f.write("Predictions: " + str(pred[:eos_pred]) + "\n")
            f.write("Labels: " + str(label[:eos_label]) + "\n")
    
    avg_anls = total_anls / total_samples  
    avg_rouge1 = total_rouge1 / total_samples
    avg_rougeL = total_rougeL / total_samples

    return avg_rouge1, avg_rougeL, avg_anls