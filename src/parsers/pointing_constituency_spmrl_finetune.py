# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime, timedelta
import torch
import torch.distributed as dist
import torch.nn as nn
from src.models import PointingConstituencySPMRLModelFinetune
from src.parsers.parser import Parser, keep_last_n_checkpoint
from src.utils import Config, Dataset, Embedding
from src.utils.logging import init_logger
from src.utils.common import bos, eos, pad, unk
from src.utils.field import ChartField, Field, RawField, SubwordField, ParsingOrderField, SubwordFieldSPMRL
from src.utils.logging import get_logger, progress_bar
from src.utils.metric import BracketMetric, Metric, SPMRL_BracketMetric, SPMRL_external_Metric
from src.utils.transform import SPMRL_Tree
from src.utils.parallel import DistributedDataParallel as DDP
from src.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau
from src.utils.spmrl_eval import evalb
logger = get_logger(__name__)


class PointingConstituencySPMRLParserFinetune(Parser):

    NAME = 'pointing-constituency-spmrl-finetune'
    MODEL = PointingConstituencySPMRLModelFinetune

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART
        self.PARSINGORDER = self.transform.PARSINGORDER
        self.total_processed = 0

    def train(self, data_path,
              buckets=32,
              batch_size=5000,
              # lr=8e-4,
              lr=2e-3,
              mu=.9,
              nu=.9,
              epsilon=1e-12,
              clip=5.0,
              decay=.75,
              decay_steps=5000,
              step_decay_factor=0.5,
              step_decay_patience=15,
              epochs=5000,
              patience=100,
              verbose=True,
              mbr=True,
              delete={'TOP', 'S1', '-NONE-', 'ROOT','VROOT'},
              equal={},
              **kwargs):
        if os.path.isfile(os.path.join(data_path,'predtrain')):
           train= os.path.join(data_path,'predtrain')
        else:
            assert os.path.isfile(os.path.join(data_path,'predtrain5k')),f'at least predtrain/predtrain5k must be there'
            train = os.path.join(data_path, 'predtrain5k')
        dev=os.path.join(data_path, 'preddev')
        test=os.path.join(data_path, 'predtest')
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)


        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Load the data")
        train = Dataset(self.transform, args.train, **args)
        dev = Dataset(self.transform, args.dev, **args)
        test = Dataset(self.transform, args.test, **args)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)

        if self.args.learning_rate_schedule=='Exponential':
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        elif self.args.learning_rate_schedule=='Plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=args.step_decay_factor,
                                               patience=args.step_decay_patience, verbose=True)


        elapsed = timedelta()
        best_e, best_metric = 1, Metric()
        best_metric_test = Metric()
        self.total_proccessed = 0
        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            loss=self._train(train.loader, warm_up=False)
            logger.info(f"{'train:':6} - loss: {loss:.4f}")
            loss, dev_metric = self._evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                dev_metric_name = '_dev_LP_{:.2f}_LR_{:.2f}_LF_{:.2f}.pt'.format(100 * best_metric.lp,
                                                                                 100 * best_metric.lr,
                                                                                 100 * best_metric.lf)
                if is_master():
                    self.save(args.path+dev_metric_name)
                logger.info(f"{t}s elapsed (saved)\n")
                keep_last_n_checkpoint(args.path + '_dev_', n=5)
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t
            if test_metric > best_metric_test:
                best_metric_test=test_metric
                test_metric_name='_test_LP_{:.2f}_LR_{:.2f}_LF_{:.2f}.pt'.format(100*best_metric_test.lp,100*best_metric_test.lr,100*best_metric_test.lf)
                if is_master():
                    self.save(args.path + test_metric_name)
                keep_last_n_checkpoint(args.path+ '_test_', n=5)
            if self.args.learning_rate_schedule == 'Plateau':
                # self.scheduler.step(best_metric.score)
                self.scheduler.step(best_metric_test.score)

            # if epoch - best_e >= args.patience:
            #     break
        loss, metric = self.load(args.path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'dev:':6} - {best_metric}")
        logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000, mbr=True,
                 delete={'TOP', 'S1', '-NONE-', 'ROOT', 'VROOT'},
                 equal={},
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
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()

        logger.info("Load the data")
        test=os.path.join(data, 'predtest')
        dataset = Dataset(self.transform, test)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start
        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")
        return super().predict(**Config().update(locals()))

    def set_lr(self,new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def schedule_lr(self,iteration):
        warmup_coeff = self.args.lr / self.args.learning_rate_warmup_steps
        iteration = iteration + 1
        if iteration <= self.args.learning_rate_warmup_steps:
            self.set_lr(iteration * warmup_coeff)

    def _train(self, loader, warm_up=False, warmup_coeff=None, start_learning_rate=0):
        self.model.train()

        bar = progress_bar(loader)
        total_loss=0
        for words, feats, trees, (spans, labels), parsingorders in bar:
            self.optimizer.zero_grad()
            if warm_up:
                self.schedule_lr(self.total_processed)
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
            self.total_processed += 1
        total_loss /= len(loader)

        return total_loss

    # @torch.no_grad()
    # def _evaluate(self, loader):
    #     self.model.eval()
    #
    #     # total_loss, metric = 0, BracketMetric()
    #     total_loss, metric = 0, SPMRL_BracketMetric()
    #
    #     for words, feats, trees, (spans, labels), parsingorders in loader:
    #         # batch_size, seq_len = words.shape
    #         # lens = words.ne(self.args.pad_index).sum(1) - 1
    #         # mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
    #         # mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
    #         # s_span, s_label = self.model(words, feats)
    #         # loss, s_span = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
    #         # chart_preds = self.model.decode(s_span, s_label, mask)
    #         # # since the evaluation relies on terminals,
    #         # # the tree should be first built and then factorized
    #         # preds = [Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
    #         #          for tree, chart in zip(trees, chart_preds)]
    #         loss = self.model.loss(words, feats, spans, labels, parsingorders)
    #         preds = self.model.decode(words, feats, beam_size=self.args.beam_size)
    #
    #         preds = [SPMRL_Tree.build(tree,
    #                        [(i, j, self.CHART.vocab.itos[label])
    #                         for i, j, label in pred])
    #                  for tree, pred in zip(trees, preds)]
    #         total_loss += loss.item()
    #         metric([SPMRL_Tree.factorize(tree, self.args.delete, self.args.equal) for tree in preds],
    #                [SPMRL_Tree.factorize(tree, self.args.delete, self.args.equal) for tree in trees])
    #     total_loss /= len(loader)
    #
    #     return total_loss, metric

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        # total_loss, metric = 0, BracketMetric()
        total_loss = 0

        final_pred = []
        gold_trees = []

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

            preds = [SPMRL_Tree.build(tree,
                                      [(i, j, self.CHART.vocab.itos[label])
                                       for i, j, label in pred])
                     for tree, pred in zip(trees, preds)]
            total_loss += loss.item()
            final_pred.extend([' '.join(str(pred).split()) for pred in preds])
            gold_trees.extend([' '.join(str(tree).split()) for tree in trees])
        with open(self.args.path + '_pred_tree.txt', 'w') as f:
            for item in final_pred:
                f.write("%s\n" % item)
            f.close()
        with open(self.args.path + '_gold_tree.txt', 'w') as f:
            for item in gold_trees:
                f.write("%s\n" % item)
            f.close()
        label_result = evalb("./src/EVALB_SPMRL", self.args.path + '_gold_tree.txt',
                       self.args.path + '_pred_tree.txt',
                       self.args.path + '_output_label.txt',label=True)
        unlabel_result = evalb("./src/EVALB_SPMRL", self.args.path + '_gold_tree.txt',
                             self.args.path + '_pred_tree.txt',
                             self.args.path + '_output_unlabel.txt', label=False)
        metric=SPMRL_external_Metric(ur=unlabel_result.recall/100.0, up=unlabel_result.precision/100.0, uf=unlabel_result.fscore/100.0,
                                     lr=label_result.recall/100.0, lp=label_result.precision/100.0, lf=label_result.fscore/100.0)

        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds, probs = {'trees': []}, []

        for words, feats, trees, (spans, labels), parsingorders in progress_bar(loader):
            # batch_size, seq_len = words.shape
            # lens = words.ne(self.args.pad_index).sum(1) - 1
            # mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # s_span, s_label = self.model(words, feats)
            # if self.args.mbr:
            #     s_span = self.model.crf(s_span, mask, mbr=True)
            # chart_preds = self.model.decode(s_span, s_label, mask)
            # preds['trees'].extend([Tree.build(tree, [(i, j, self.CHART.vocab[label]) for i, j, label in chart])
            #                        for tree, chart in zip(trees, chart_preds)])
            preds_ = self.model.decode(words, feats, beam_size=1)
            preds['trees'].extend( [SPMRL_Tree.build(tree,
                           [(i, j, self.CHART.vocab.itos[label])
                            for i, j, label in pred])
                     for tree, pred in zip(trees, preds_)])
            # if self.args.prob:
            #     probs.extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span.unbind())])
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, data_path='',**kwargs):
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
        if os.path.isfile(os.path.join(data_path,'predtrain')):
           train= os.path.join(data_path,'predtrain')
        else:
            assert os.path.isfile(os.path.join(data_path,'predtrain5k')),f'at least predtrain/predtrain5k must be there'
            train = os.path.join(data_path, 'predtrain5k')
        dev=os.path.join(data_path, 'preddev')
        test=os.path.join(data_path, 'predtest')

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
            FEAT = SubwordFieldSPMRL('chars', pad=pad, unk=unk, bos=bos, eos=eos, fix_len=args.fix_len)
        elif args.feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            FEAT = SubwordFieldSPMRL('bert',
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
            transform = SPMRL_Tree(WORD=(WORD, FEAT), TREE=TREE, CHART=CHART, PARSINGORDER=PARSINGORDER)
        else:
            transform = SPMRL_Tree(WORD=WORD, POS=FEAT, TREE=TREE, CHART=CHART, PARSINGORDER=PARSINGORDER)

        train = Dataset(transform, args.train,
                        binarize_direction=args.binarize_direction,
                        dummy_label_manipulating=args.dummy_label_manipulating)
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
