#!/usr/bin/env python2

"""
Evaluations: BLEU-1, BLEU-2, BLEU-3, BLEU-4, Meteor, ROUGE, CIDEr.

The underlying codes follow `pycocoevalcap`:
https://github.com/tylin/coco-caption/tree/master/pycocoevalcap

You have to download it and put the directory 'pycocoevalcap' in CWD.

`pycocoevalcap` requires *python2* and java.
"""

from codecs import open

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class EvalCaptions():
    def __init__(self, gt_filename, res_filename):
        self.gt_filename = gt_filename
        self.res_filename = res_filename
        self.gts = self.format_gt_file(gt_filename)
        self.res = self.format_res_file(res_filename)
        self.score = {}
        self.scores = {}


    def format_gt_file(self, gt_filename):
        """Read and format the ground truth captions file."""
        gts = {}
        index = 8001
        with open(gt_filename, 'rb', encoding='utf8') as f:
            first_line = f.readline().lstrip(u'\ufeff').rstrip()
            if first_line.isdigit():
                index = int(first_line)
            gts[index] = []

            for line in f:
                line_s = line.rstrip()
                if line_s.isdigit() and int(line_s) == index + 1:
                    index += 1
                    gts[index] = []
                else:
                    caption = ' '.join(list(line_s)).encode('utf8')
                    gts[index].append({'caption': caption})
        return gts


    def format_res_file(self, res_filename):
        """Read and format the predicted result captions file."""
        res = {}
        with open(res_filename, 'rb', encoding='utf8') as f:
            for line in f:
                index, caption = line.rstrip().split(' ', 1)
                index = int(index)
                caption = caption.encode('utf8')
                res[index] = [{'caption': caption}]
        return res


    def evaluate(self):
        """Perform evaluations: BLEU-1, BLEU-2, BLEU-3, BLEU-4, Meteor, ROUGE, CIDEr"""
        # tokenization
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.gts)
        res = tokenizer.tokenize(self.res)
        # setup scorers
        scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Meteor(),'METEOR'),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr')
        ]

        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.score[m] = sc
                    self.scores[m] = scs
                    print '%s: %0.3f' % (m, sc)
            else:
                self.score[method] = score
                self.scores[method] = scores
                print '%s: %0.3f' % (method, score)


def main(gt_filename, res_filename):
    evaluator = EvalCaptions(gt_filename, res_filename)
    evaluator.evaluate()


if __name__ == '__main__':
    from sys import argv

    if len(argv) != 3:
        print 'Usage: %s <path/to/ground/truth.txt> <path/to/predict/results.txt>' % argv[0]
        exit()

    main(argv[1], argv[2])
