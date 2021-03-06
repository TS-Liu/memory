from collections import OrderedDict

import numpy
import torch
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 cuda=False,
                 beam_trace=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[]):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, word_batch, data, lists):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object


        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        tgt_vocab = self.fields["tgt"].vocab

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([tgt_vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=tgt_vocab.stoi[onmt.io.PAD_WORD],
                                    eos=tgt_vocab.stoi[onmt.io.EOS_WORD],
                                    bos=tgt_vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        src_lines = word_batch

        N = 1
        M = 1

        ss_contexts=[]
        for src_line in src_lines :
            src_words = [w.encode('utf-8') for w in list(src_line.src)]
            s_num = 0
            s_contexts = []
            for words in src_words:

                s_context = [''] * (M * 2 + 1)
                if s_num < M and s_num + M + 1 <= len(src_words):
                    s_context[M * 2 - s_num - M:] = src_words[0:s_num + M + 1]
                    s_context[:M * 2 - s_num - M] = ['<s>'] * (M * 2 - s_num - M)
                elif s_num - M >= 0 and s_num > len(src_words) - M - 1:
                    s_context[0:len(src_words) - s_num + M] = src_words[s_num - M:len(src_words)]
                    s_context[len(src_words) - s_num + M:] = ['</s>'] * (M * 2 + 1 - len(src_words) + s_num - M)
                elif s_num - M >= 0 and s_num + M + 1 <= len(src_words):
                    s_context = src_words[s_num - M:s_num + M + 1]
                else:
                    s_context[M * 2 - s_num - M:len(src_words) - s_num + M] = src_words
                    s_context[len(src_words) - s_num + M:] = ['</s>'] * (M * 2 + 1 - len(src_words) + s_num - M)
                    s_context[:M * 2 - s_num - M] = ['<s>'] * (M * 2 - s_num - M)

                s_contexts.append(s_context)
                s_num += 1
            ss_contexts.append(s_contexts)




        if data_type == 'text':
            _, src_lengths = batch.src

        enc_states, memory_bank, src_embeddings = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
                                        src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None
        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        dec_states.repeat_beam_size_times(beam_size)


        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            tgt_vocab = self.fields["tgt"].vocab
            beam_tokens=[]
            for tok in list(inp[0].data.cpu().numpy()):
                beam_tokens.append(tgt_vocab.itos[tok].encode('utf-8'))

            inp = inp.unsqueeze(2)

            src_memorys = []
            tgt_memorys = []
            src_ms = []
            tgt_ms = []

            beam_tokens=numpy.array(beam_tokens).reshape(5,-1).transpose()
            i=0

            for contexts in ss_contexts:
                tokens = beam_tokens[i]
                src_memory = []
                tgt_memory = []
                src_m = []
                tgt_m = []
                for token in tokens:
                    context_wordt = OrderedDict()
                    for context in contexts :
                        if str(context) in lists:
                            if str([token]) in lists[str(context)]:
                                for ws in lists[str(context)][str([token])].keys():
                                    if str([context, ws]) not in context_wordt:
                                        context_wordt[str([context, ws])] = lists[str(context)][str([token])][ws]
                                    else:
                                        context_wordt[str([context, ws])] = context_wordt[str([context, ws])] + \
                                                                            lists[str(context)][str([token])][ws]

                    context_wordt_sorted = sorted(context_wordt.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

                    if len(context_wordt_sorted) > 0:
                        context_wordt_sorted = context_wordt_sorted[0:3]

                        for c_w in context_wordt_sorted:
                            w = eval(c_w[0])[1]
                            src_memory.append(eval(c_w[0])[0])
                            tgt_memory.append([token])
                            src_m.append(eval(c_w[0])[0][1])
                            tgt_m.append(w)

                        if len(context_wordt_sorted) < 3:
                            for nw in range(3 - len(context_wordt_sorted)):
                                src_memory.append(['<blank>'] * (M * 2 + 1))
                                tgt_memory.append(['<blank>'] * N)
                                src_m.append('<blank>')
                                tgt_m.append('<blank>')
                    else:
                        for nw in range(3):
                            src_memory.append(['<blank>'] * (M * 2 + 1))
                            tgt_memory.append(['<blank>'] * N)
                            src_m.append('<blank>')
                            tgt_m.append('<blank>')
                src_memorys.append(src_memory)
                tgt_memorys.append(tgt_memory)
                src_ms.append(src_m)
                tgt_ms.append(tgt_m)
                i+=1

            src_vocab = self.fields["src"].vocab
            tgt_vocab = self.fields["tgt"].vocab
            src_memorys = [[[src_vocab.stoi[memorysx.decode('utf8')] for memorysx in memorysxx ] for memorysxx in memorysxxx ] for memorysxxx in src_memorys]
            src_ms = [[src_vocab.stoi[msx.decode('utf8')] for msx in msxx ] for msxx in src_ms ]
            tgt_memorys = [[[tgt_vocab.stoi[memorysx.decode('utf8')] for memorysx in memorysxx] for memorysxx in memorysxxx] for memorysxxx in tgt_memorys]
            tgt_ms = [[tgt_vocab.stoi[msx.decode('utf8')] for msx in msxx] for msxx in tgt_ms]

            src_memorys = Variable(torch.LongTensor(numpy.array(src_memorys)).cuda().view(-1,5,9,1).transpose(0,2).contiguous()).view(9,-1,1)
            tgt_memorys = Variable(torch.LongTensor(numpy.array(tgt_memorys)).cuda().view(-1,5,3,1).transpose(0,2).contiguous()).view(3,-1,1)
            src_ms = Variable(torch.LongTensor(numpy.array(src_ms)).cuda().view(-1,5,3,1).transpose(0,2).contiguous()).view(3,-1,1)
            tgt_ms = Variable(torch.LongTensor(numpy.array(tgt_ms)).cuda().view(-1,5,3,1).transpose(0,2).contiguous()).view(3,-1,1)


            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, src_memorys, tgt_memorys, src_ms, tgt_ms, memory_bank, src_embeddings, dec_states,
                memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data
                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"] = self._run_target(batch, data)
        ret["batch"] = batch
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, _ = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores
        return gold_scores
