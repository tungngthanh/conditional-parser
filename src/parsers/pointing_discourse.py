import os,glob
from datetime import datetime, timedelta
import torch
import torch.distributed as dist
import torch.nn as nn
from src.models import PointingDiscourseModel
from src.parsers.parser import Parser, keep_last_n_checkpoint
from src.utils import Config, Dataset, Embedding
from src.utils.logging import init_logger
from src.utils.common import bos, eos, pad, unk
from src.utils.field import ChartDiscourseField, Field, RawField, SubwordField, ParsingOrderField
from src.utils.logging import get_logger, progress_bar
from src.utils.metric import DiscourseMetric, Metric
from src.utils.transform import DiscourseTree
from src.utils.parallel import DistributedDataParallel as DDP
from src.utils.parallel import is_master
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau
logger = get_logger(__name__)


class PointingDiscourseParser(Parser):

    NAME = 'pointing-discourse'
    MODEL = PointingDiscourseModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.feat in ('char', 'bert'):
            self.WORD, self.FEAT = self.transform.WORD
        else:
            self.WORD, self.FEAT = self.transform.WORD, self.transform.POS
        self.EDU_BREAK = self.transform.EDU_BREAK
        self.GOLD_METRIC = self.transform.GOLD_METRIC
        # self.TREE = self.transform.TREE
        self.CHART = self.transform.CHART
        self.PARSINGORDER = self.transform.PARSINGORDER

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
              **kwargs):
        train = {'sentences': os.path.join(data_path, "Training_InputSentences.pickle"),
                 'edu_break': os.path.join(data_path, "Training_EDUBreaks.pickle"),
                 'golden_metric': os.path.join(data_path, "Training_GoldenLabelforMetric.pickle")}
        # train = {'sentences': os.path.join(data_path, "Testing_InputSentences.pickle"),
        #         'edu_break': os.path.join(data_path, "Testing_EDUBreaks.pickle"),
        #         'golden_metric': os.path.join(data_path, "Testing_GoldenLabelforMetric.pickle")}
        test = {'sentences': os.path.join(data_path, "Testing_InputSentences.pickle"),
                 'edu_break': os.path.join(data_path, "Testing_EDUBreaks.pickle"),
                 'golden_metric': os.path.join(data_path, "Testing_GoldenLabelforMetric.pickle")}

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        if dist.is_initialized():
            args.batch_size = args.batch_size // dist.get_world_size()
        logger.info("Load the data")
        train = Dataset(self.transform, args.train, **args)
        # dev = Dataset(self.transform, args.dev)
        test = Dataset(self.transform, args.test)
        train.build(args.batch_size, args.buckets, True, dist.is_initialized())
        # dev.build(args.batch_size, args.buckets)
        test.build(args.batch_size, args.buckets)
        # logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        logger.info(f"\n{'train:':6} {train}\n{'test:':6} {test}\n")

        logger.info(f"{self.model}\n")
        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)
        self.optimizer = Adam(self.model.parameters(),
                              args.lr,
                              (args.mu, args.nu),
                              args.epsilon)
        if self.args.learning_rate_schedule == 'Exponential':
            self.scheduler = ExponentialLR(self.optimizer, args.decay ** (1 / args.decay_steps))
        elif self.args.learning_rate_schedule == 'Plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=args.step_decay_factor,
                                               patience=args.step_decay_patience, verbose=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()
        # best_metric_test = Metric()
        for epoch in range(1, args.epochs + 1):
            start = datetime.now()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            loss = self._train(train.loader)
            logger.info(f"{'train:':6} - loss: {loss:.4f}")
            # loss, dev_metric = self._evaluate(dev.loader)
            # logger.info(f"{'dev:':6} - loss: {loss:.4f} - {dev_metric}")
            loss, test_metric = self._evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {loss:.4f} - {test_metric}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if test_metric > best_metric:
                best_e, best_metric = epoch, test_metric
                # dev_metric_name = '_dev_LP_{:.2f}_LR_{:.2f}_LF_{:.2f}.pt'.format(100 * best_metric.lp,
                #                                                                  100 * best_metric.lr,
                #                                                                  100 * best_metric.lf)
                test_metric_name = '_test_UF_{:.2f}_NF_{:.2f}_RF_{:.2f}.pt'.format(100 * best_metric.uf,
                                                                                   100 * best_metric.nf,
                                                                                   100 * best_metric.rf)
                if is_master():
                    # self.save(args.path + dev_metric_name)
                    self.save(args.path + test_metric_name)
                logger.info(f"{t}s elapsed (saved)\n")
                keep_last_n_checkpoint(args.path + '_test_', n=5)
            else:
                logger.info(f"{t}s elapsed\n")
            elapsed += t

            if self.args.learning_rate_schedule == 'Plateau':
                self.scheduler.step(best_metric.score)


            # if epoch - best_e >= args.patience:
            #     break
        # loss, metric = self.load(args.path)._evaluate(test.loader)

        logger.info(f"Epoch {best_e} saved")
        logger.info(f"{'test:':6} - {best_metric}")
        # logger.info(f"{'test:':6} - {metric}")
        logger.info(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

    def evaluate(self, data, buckets=8, batch_size=5000,
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

    def predict(self, data, pred=None, buckets=8, batch_size=5000, prob=False, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()

        logger.info("Load the data")
        test = {'sentences': os.path.join(data, "Testing_InputSentences.pickle"),
                 'edu_break': os.path.join(data, "Testing_EDUBreaks.pickle"),
                 'golden_metric': os.path.join(data, "Testing_GoldenLabelforMetric.pickle")}
        dataset = Dataset(self.transform, test)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Make predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        import pickle
        with open('/mnt/StorageDevice/Projects/New_pointing_encoder_Decoder_Parsing/exp/ptb.pointing.discourse.char/predic_file','wb') as f:
            pickle.dump(preds, f)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None:
            logger.info(f"Save predicted results to {pred}")
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)
        total_loss=0
        for words, feats, edu_break, golden_metric, (spans, labels), parsing_orders in bar:
            self.optimizer.zero_grad()

            # batch_size, seq_len = words.shape
            # lens = words.ne(self.args.pad_index).sum(1) - 1
            # mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            # mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
            # s_span, s_label = self.model(words, feats)
            # loss, _ = self.model.loss(s_span, s_label, spans, labels, mask, self.args.mbr)
            loss = self.model.loss(words, feats, spans, labels, parsing_orders)
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

        total_loss, metric = 0, DiscourseMetric()


        for words, feats, edu_break, golden_metric, (spans, labels), parsing_orders in loader:
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
            loss = self.model.loss(words, feats, spans, labels, parsing_orders)
            preds = self.model.decode(words, feats)

            # preds = [Tree.build(tree,
            #                [(i, j, self.CHART.vocab.itos[label])
            #                 for i, j, label in pred])
            #          for tree, pred in zip(trees, preds)]
            preds=[[(i, k, j, self.CHART.vocab.itos[label])
                   for i, k, j, label in pred]
                   for pred in preds]
            total_loss += loss.item()
            metric([DiscourseTree.build(tree)
                    for tree in preds],
                   [DiscourseTree.build_gold(_edu_break, _golden_metric)
                    for _edu_break, _golden_metric in zip(edu_break,golden_metric)])
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {'trees': [], 'gold_trees':[],'edu_break':[], 'golden_metric':[]}

        # for words, feats in progress_bar(loader):
        for words, feats, edu_break, golden_metric, (spans, labels), parsing_orders in loader:
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
            preds['trees'].extend( [DiscourseTree.build(
                [(i, k, j, self.CHART.vocab.itos[label])
                 for i, k, j, label in pred])
                for pred in preds_])
            preds['gold_trees'].extend([DiscourseTree.build_gold(_edu_break, _golden_metric)
                    for _edu_break, _golden_metric in zip(edu_break,golden_metric)])
            preds['edu_break'].extend(edu_break)
            preds['golden_metric'].extend(golden_metric)
            # if self.args.prob:
            #     probs.extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span.unbind())])
        # if self.args.prob:
        #     preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, data_path='', **kwargs):
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
        train = {'sentences': os.path.join(data_path, "Training_InputSentences.pickle"),
                 'edu_break': os.path.join(data_path, "Training_EDUBreaks.pickle"),
                 'golden_metric': os.path.join(data_path, "Training_GoldenLabelforMetric.pickle")}
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

        EDU_BREAK = RawField('edu_break')
        GOLD_METRIC = RawField('golden_metric')
        # CHART = ChartDiscourseField('charts_discourse', pad=pad)
        CHART = ChartDiscourseField('charts_discourse')
        PARSINGORDER = ParsingOrderField('parsingorder')
        if args.feat in ('char', 'bert'):
            transform = DiscourseTree(WORD=(WORD, FEAT), EDU_BREAK=EDU_BREAK, GOLD_METRIC=GOLD_METRIC, CHART=CHART, PARSINGORDER=PARSINGORDER)
        # else:
        #     transform = DiscourseTree(WORD=WORD, EDU_BREAK=EDU_BREAK, GOLD_METRIC=GOLD_METRIC, CHART=CHART, PARSINGORDER=PARSINGORDER)

        train = Dataset(transform, args.train)
        # WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk) if args.embed else None))
        WORD.build(train, args.min_freq)
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
