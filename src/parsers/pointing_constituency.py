# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from src.models import PointingConstituencyModel
from src.parsers.parser import Parser
from src.utils import Config, Dataset, Embedding
from src.utils.common import bos, eos, pad, unk
from src.utils.field import ChartField, Field, RawField, SubwordField, ParsingOrderField
from src.utils.logging import get_logger, progress_bar, init_logger
from src.utils.metric import BracketMetric
from src.utils.transform import Tree

logger = get_logger(__name__)


class PointingConstituencyParser(Parser):
    """
    The implementation of Pointing Constituency Parser.

    """

    NAME = 'pointing-constituency'
    MODEL = PointingConstituencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART
        self.PARSINGORDER = self.transform.PARSINGORDER

    def train(self, train, dev, test, buckets=32, batch_size=5000, mbr=True,
              delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
              equal={'ADVP': 'PRT'},
              verbose=True,
              **kwargs):
        """
        Args:
            train, dev, test (list[list] or str):
                the train/dev/test data, both list of instances and filename are allowed.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            mbr (bool):
                If ``True``, performs mbr decoding. Default: ``True``.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000, mbr=True,
                 delete={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''},
                 equal={'ADVP': 'PRT'},
                 verbose=True,
                 **kwargs):
        """
        Args:
            data (str):
                The data to be evaluated.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            mbr (bool):
                If ``True``, performs mbr decoding. Default: ``True``.
            delete (set[str]):
                A set of labels that will not be taken into consideration during evaluation.
                Default: {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}.
            equal (dict[str, str]):
                The pairs in the dict are considered equivalent during evaluation.
                Default: {'ADVP': 'PRT'}.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, mbr=True, verbose=True, **kwargs):
        # args = self.args.update(locals())
        # init_logger(logger, verbose=args.verbose)
        #
        # self.transform.eval()
        #
        # logger.info("Load the data")
        # test=os.path.join(data, 'predtest')
        # dataset = Dataset(self.transform, test)
        # dataset.build(args.batch_size, args.buckets)
        # logger.info(f"\n{dataset}")
        #
        # logger.info("Make predictions on the dataset")
        # start = datetime.now()
        # preds = self._predict(dataset.loader)
        # elapsed = datetime.now() - start
        # for name, value in preds.items():
        #     setattr(dataset, name, value)
        # if pred is not None:
        #     logger.info(f"Save predicted results to {pred}")
        #     self.transform.save(pred, dataset.sentences)
        # logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")
        #
        # return dataset
        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)
        total_loss=0
        for words, feats, trees, (spans, labels), parsingorders in bar:
            self.optimizer.zero_grad()

            # batch_size, seq_len = words.shape
            # lens = words.ne(self.args.pad_index).sum(1) - 1
            # mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # s_span, s_label = self.model(words, feats)
            # loss, _ = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
            loss = self.model.loss(words, feats, spans, labels, parsingorders)
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            if self.args.learning_rate_schedule=='Exponential':
                self.scheduler.step()
            if self.args.learning_rate_schedule == 'Exponential':
                bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
            elif self.args.learning_rate_schedule == 'Plateau':
                bar.set_postfix_str(f"lr: {self.scheduler.optimizer.param_groups[0]['lr']:.4e} - loss: {loss:.4f}")
                # bar.set_postfix_str(f"loss: {loss:.4f}")
        total_loss /= len(loader)

        return total_loss

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, BracketMetric()

        for words, feats, trees, (spans, labels), parsingorders in loader:
            # batch_size, seq_len = words.shape
            # lens = words.ne(self.args.pad_index).sum(1) - 1
            # mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # s_span, s_label = self.model(words, feats)
            # loss, s_span = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
            # chart_preds = self.model.decode(s_span, s_label, mask)
            # # since the evaluation relies on terminals,
            # # the tree should be first built and then factorized
            # preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
            #          for tree, chart in zip(trees, chart_preds)]
            loss = self.model.loss(words, feats, spans, labels, parsingorders)
            preds = self.model.decode(words, feats, beam_size=self.args.beam_size)

            preds = [Tree.build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            metric([Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
                   [Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds, probs = {'trees': []}, []

        for words, feats, trees, (spans, labels), parsingorders in progress_bar(loader):
            preds_ = self.model.decode(words, feats, beam_size=1)
            preds['trees'].extend( [Tree.build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds_)])
            # if self.args.prob:
            #     probs.extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span.unbind())])
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        """
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The created parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Build the fields")
        WORD = Field('words', pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
        if args.feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.cls_token or tokenizer.cls_token,
                                eos=tokenizer.sep_token or tokenizer.sep_token,
                                fix_len=args.fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos, eos=eos)
        TREE = RawField('trees')
        CHART = ChartField('charts')
        PARSINGORDER = ParsingOrderField('parsingorder')
        if args.feat in ('char', 'bert'):
            transform = Tree(WORD=(WORD, FEAT), TREE=TREE, CHART=CHART, PARSINGORDER=PARSINGORDER)
        else:
            transform = Tree(WORD=WORD, POS=FEAT, TREE=TREE, CHART=CHART, PARSINGORDER=PARSINGORDER)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        FEAT.build(train)
        CHART.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_feats': len(FEAT.vocab),
            'n_labels': len(CHART.vocab),
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'eos_index': WORD.eos_index,
            'feat_pad_index': FEAT.pad_index
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
