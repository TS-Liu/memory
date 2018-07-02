#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import pickle
from collections import OrderedDict

import torch

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
from multiprocessing import Pool

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    path = os.path.split(os.path.realpath(__file__))[0]
    print()
    res = subprocess.check_output(
        "perl %s/tools/multi-bleu.perl %s < %s"
        % (path, opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    path = os.path.split(os.path.realpath(__file__))[0]
    res = subprocess.check_output(
        "python %s/tools/test_rouge.py -r %s -c %s"
        % (path, opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())

def func(i,N,M,src_lines, trg_lines, align_lines):
    print("start"+str(i))
    align = OrderedDict()
    for lines, linet, linea in zip(src_lines, trg_lines, align_lines):
        words_src = lines.strip().split()
        words_trg = linet.strip().split()
        a_s_t = linea.strip().split()
        for a in a_s_t :
            a = a.split('-')
            t_num = int(a[1])
            s_num = int(a[0])
            t = words_trg[t_num]
            s = words_src[s_num]
            t_n_gram=['<s>']*(N)
            s_context=['']*(M*2+1)
            if t_num < N :
                t_n_gram[N-t_num:] = words_trg[0:t_num]
            else :
                t_n_gram = words_trg[t_num-N:t_num]
            if s_num < M and s_num+M+1 <= len(words_src):
                s_context[M*2-s_num-M:] = words_src[0:s_num+M+1]
                s_context[:M*2-s_num-M] = ['<s>']*(M*2-s_num-M)
            elif s_num-M >= 0 and s_num > len(words_src)-M-1 :
                s_context[0:len(words_src)-s_num+M] = words_src[s_num-M:len(words_src)]
                s_context[len(words_src)-s_num+M:] = ['</s>']*(M*2+1-len(words_src)+s_num-M)
            elif s_num-M >= 0 and s_num+M+1 <= len(words_src) :
                s_context = words_src[s_num-M:s_num+M+1]
            else :
                s_context[M*2-s_num-M:len(words_src)-s_num+M] = words_src
                s_context[len(words_src)-s_num+M:] = ['</s>']*(M*2+1-len(words_src)+s_num-M)
                s_context[:M*2-s_num-M] = ['<s>']*(M*2-s_num-M)
            if str([t,s,t_n_gram,s_context]) not in align.keys() :
                align[str([t,s,t_n_gram,s_context])] = 0
            align[str([t,s,t_n_gram,s_context])]+=1
    print("end"+str(i))
    return align
def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)



    # N = 1
    # M = 1
    # src_file = open(opt.train_src, 'r')
    # trg_file = open(opt.train_tgt, 'r')
    #
    # align_file = open(opt.train_align, 'r')
    #
    # src_lines = src_file.readlines()
    # trg_lines = trg_file.readlines()
    # align_lines = align_file.readlines()
    #
    # align = OrderedDict()
    # pool = Pool()
    # result = []
    # for i in range(10000):
    #     result.append(pool.apply_async(func, args=(
    #     i, N, M, src_lines[125 * i:125 * (i + 1)], trg_lines[125 * i:125 * (i + 1)],
    #     align_lines[125 * i:125 * (i + 1)])))
    # pool.close()
    # pool.join()
    #
    # for i in result:
    #     ddict = i.get()
    #     for k, v in ddict.items():
    #         if k not in align:
    #             align[k] = v
    #         else:
    #             align[k] = v + align[k]
    #
    # align_sorted = sorted(align.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    # print(len(align_sorted))
    #
    # k = 0
    # lists = OrderedDict()
    # for align in align_sorted:
    #     pairs = eval(align[0])
    #     y = pairs[0]
    #     ngram = pairs[2]
    #     context = pairs[3]
    #     if align[1] > 0:
    #         if str(ngram) not in lists:
    #             value = OrderedDict()
    #             xy = OrderedDict()
    #             xy[y] = align[1]
    #             value[str(context)] = xy
    #             lists[str(ngram)] = value
    #         else:
    #             if str(context) not in lists[str(ngram)]:
    #                 xy = OrderedDict()
    #                 xy[y] = align[1]
    #                 lists[str(ngram)][str(context)] = xy
    #             else:
    #                 if y not in lists[str(ngram)][str(context)]:
    #                     lists[str(ngram)][str(context)][y] = align[1]
    #                 else:
    #                     lists[str(ngram)][str(context)][y] = align[1] + lists[str(ngram)][str(context)][y]
    #         k += 1
    # print(k)
    pkl_file = open(opt.lists, 'rb')
    lists = pickle.load(pkl_file)
    pkl_file.close()

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)
    model.cuda()
    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.TestOrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)
    translator = onmt.translate.Translator(
        model, fields,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        copy_attn=model_opt.copy_attn,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length,
        stepwise_penalty=opt.stepwise_penalty,
        block_ngram_repeat=opt.block_ngram_repeat,
        ignore_when_blocking=opt.ignore_when_blocking)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0




    for batch, word_batch in data_iter:
        batch_data = translator.translate_batch(batch, word_batch, data, lists)
        translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent) + 1

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))

    _report_score('PRED', pred_score_total, pred_words_total)
    if opt.tgt:
        _report_score('GOLD', gold_score_total, gold_words_total)
        if opt.report_bleu:
            _report_bleu()
        if opt.report_rouge:
            _report_rouge()

    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))


if __name__ == "__main__":
    main()
