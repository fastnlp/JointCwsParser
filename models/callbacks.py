
from fastNLP.core.callback import Callback
import torch
from torch import nn

class OptimizerCallback(Callback):
    def __init__(self, optimizer, scheduler, update_every=4):
        super().__init__()

        self._optimizer = optimizer
        self.scheduler = scheduler
        self._update_every = update_every

    def on_backward_end(self):
        if self.step % self._update_every==0:
            # nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 5)
            # self._optimizer.step()
            self.scheduler.step()
            # self.model.zero_grad()


class DevCallback(Callback):
    def __init__(self, tester, metric_key='u_f1'):
        super().__init__()
        self.tester = tester
        setattr(tester, 'verbose', 0)

        self.metric_key = metric_key

        self.record_best = False
        self.best_eval_value = 0
        self.best_eval_res = None

        self.best_dev_res = None # 存取dev的表现

    def on_valid_begin(self):
        eval_res = self.tester.test()
        metric_name = self.tester.metrics[0].__class__.__name__
        metric_value = eval_res[metric_name][self.metric_key]
        if metric_value>self.best_eval_value:
            self.best_eval_value = metric_value
            self.best_epoch = self.trainer.epoch
            self.record_best = True
            self.best_eval_res = eval_res
        self.test_eval_res = eval_res
        eval_str = "Epoch {}/{}. \n".format(self.trainer.epoch, self.n_epochs) + \
                   self.tester._format_eval_results(eval_res)
        self.pbar.write(eval_str)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if self.record_best:
            self.best_dev_res = eval_result
            self.record_best = False
        if is_better_eval:
            self.best_dev_res_on_dev = eval_result
            self.best_test_res_on_dev = self.test_eval_res
            self.dev_epoch = self.epoch

    def on_train_end(self):
        print("Got best test performance in epoch:{}\n Test: {}\n Dev:{}\n".format(self.best_epoch,
                                                            self.tester._format_eval_results(self.best_eval_res),
                                                            self.tester._format_eval_results(self.best_dev_res)))
        print("Got best dev performance in epoch:{}\n Test: {}\n Dev:{}\n".format(self.dev_epoch,
                                                            self.tester._format_eval_results(self.best_test_res_on_dev),
                                                            self.tester._format_eval_results(self.best_dev_res_on_dev)))


from fastNLP import Callback, Tester, DataSet


class EvaluateCallback(Callback):
    """
    通过使用该Callback可以使得Trainer在evaluate dev之外还可以evaluate其它数据集，比如测试集。每一次验证dev之前都会先验证EvaluateCallback
    中的数据。
    """

    def __init__(self, data=None, tester=None):
        """
        :param ~fastNLP.DataSet,Dict[~fastNLP.DataSet] data: 传入DataSet对象，会使用Trainer中的metric对数据进行验证。如果需要传入多个
            DataSet请通过dict的方式传入。
        :param ~fastNLP.Tester,Dict[~fastNLP.DataSet] tester: Tester对象, 通过使用Tester对象，可以使得验证的metric与Trainer中
            的metric不一样。
        """
        super().__init__()
        self.datasets = {}
        self.testers = {}
        self.best_test_metric_sofar = 0
        self.best_test_sofar = None
        self.best_test_epoch = 0
        self.best_dev_test = None
        self.best_dev_epoch = 0
        if tester is not None:
            if isinstance(tester, dict):
                for name, test in tester.items():
                    if not isinstance(test, Tester):
                        raise TypeError(f"{name} in tester is not a valid fastNLP.Tester.")
                    self.testers['tester-' + name] = test
            if isinstance(tester, Tester):
                self.testers['tester-test'] = tester
            for tester in self.testers.values():
                setattr(tester, 'verbose', 0)

        if isinstance(data, dict):
            for key, value in data.items():
                assert isinstance(value, DataSet), f"Only DataSet object is allowed, not {type(value)}."
            for key, value in data.items():
                self.datasets['data-' + key] = value
        elif isinstance(data, DataSet):
            self.datasets['data-test'] = data
        elif data is not None:
            raise TypeError("data receives dict[DataSet] or DataSet object.")

    def on_train_begin(self):
        if len(self.datasets) > 0 and self.trainer.dev_data is None:
            raise RuntimeError("Trainer has no dev data, you cannot pass extra DataSet to do evaluation.")

        if len(self.datasets) > 0:
            for key, data in self.datasets.items():
                tester = Tester(data=data, model=self.model,
                                batch_size=self.trainer.kwargs.get('dev_batch_size', self.batch_size),
                                metrics=self.trainer.metrics, verbose=0,
                                use_tqdm=self.trainer.test_use_tqdm)
                self.testers[key] = tester

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        if len(self.testers) > 0:
            for idx, (key, tester) in enumerate(self.testers.items()):
                try:
                    eval_result = tester.test()
                    if idx == 0:
                        indicator, indicator_val = _check_eval_results(eval_result)
                        if indicator_val>self.best_test_metric_sofar:
                            self.best_test_metric_sofar = indicator_val
                            self.best_test_epoch = self.epoch
                            self.best_test_sofar = eval_result
                    if better_result:
                        self.best_dev_test = eval_result
                        self.best_dev_epoch = self.epoch
                    self.logger.info("EvaluateCallback evaluation on {}:".format(key))
                    self.logger.info(tester._format_eval_results(eval_result))
                except Exception as e:
                    self.logger.error("Exception happens when evaluate on DataSet named `{}`.".format(key))
                    raise e

    def on_train_end(self):
        if self.best_test_sofar:
            self.logger.info("Best test performance(may not correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_test_sofar, self.best_test_epoch))
        if self.best_dev_test:
            self.logger.info("Best test performance(correspond to the best dev performance):{} achieved at Epoch:{}.".format(self.best_dev_test, self.best_dev_epoch))


def _check_eval_results(metrics, metric_key=None):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        metric_dict = list(metrics.values())[0]  # 取第一个metric

        if metric_key is None:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
            indicator = metric_key
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator, indicator_val
