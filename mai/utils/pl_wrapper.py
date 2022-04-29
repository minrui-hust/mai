import itertools
import math
import os
import tempfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset

from mai.data.datasets import Sample
from mai.utils import FI


class PlWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # model
        self.train_model = FI.create(config['model']['train'])
        self.train_model.set_train()

        self.util_models = {}  # use normal dict to avoid register parameters
        self.util_models['eval_model'] = FI.create(config['model']['eval'])
        self.util_models['eval_model'].set_eval()

        self.util_models['export_model'] = FI.create(config['model']['export'])
        self.util_models['export_model'].set_export()

        # codec
        self.train_codec = FI.create(config['codec']['train'])
        self.train_codec.set_train()

        self.eval_codec = FI.create(config['codec']['eval'])
        self.eval_codec.set_eval()

        self.export_codec = FI.create(config['codec']['export'])
        self.export_codec.set_export()

        # collater
        self.train_collater = self.train_codec.collater()
        self.eval_collater = self.eval_codec.collater()
        self.export_collater = self.export_codec.collater()

        # check if we are doing tensorrt eval
        self.trt_model = None
        if 'trt_engine' in config:
            trt_engine = config['trt_engine']
            trt_plugin = config.get('trt_plugin', None)
            trt_sequential = config.get('trt_sequential', False)
            from mai.utils.trt_model import TrtModel
            self.trt_model = TrtModel(trt_engine, trt_plugin, trt_sequential)
            self.eval_codec.set_trt()

    def forward(self, *input):
        # forward is used for export
        output = self.export_model(*input)
        pred = self.export_codec.decode(output)
        return pred

    def training_step(self, batch, batch_idx):
        output = self.train_model(batch)
        loss_dict = self.train_codec.loss(output, batch)

        # log loss
        for name, value in loss_dict.items():
            if name != 'loss':
                value.detach_()
            self.log(f'train/{name}', value)

        return loss_dict

    def training_epoch_end(self, epoch_output):
        pass

    def validation_step(self, batch, batch_idx, *args):
        output = self.eval_model(batch)

        # check if should evaluate and log loss
        if self.eval_evaluate_loss:
            loss_dict = self.eval_codec.loss(output, batch)
            if self.eval_log_loss:
                for name, value in loss_dict.items():
                    self.log(f'eval/{name}', value,
                             batch_size=batch['_info_']['size'])

        # construct batch interested
        batch_interest = Sample(_info_=batch['_info_'])
        for key in self.eval_interest_set:
            if key == 'pred':
                batch_interest[key] = self.eval_codec.decode(output, batch)
            elif key == 'output':
                batch_interest[key] = output
            elif key == 'loss':
                for loss_name in loss_dict.keys():
                    loss_dict[loss_name] = loss_dict[loss_name].detach(
                    ).cpu().item()
                batch_interest[key] = loss_dict
            elif key in batch:
                batch_interest[key] = batch[key]
            else:
                pass

        # decollate to sample_list
        sample_list = self.eval_collater.decollate(batch_interest)

        # do callback on samples
        if self.eval_step_hook is not None and sample_list:
            for sample in sample_list:
                if isinstance(self.eval_step_hook, list):
                    for hook in self.eval_step_hook:
                        hook(sample, self)
                else:
                    self.eval_step_hook(sample, self)

        # output for epoch end
        eval_step_out = []
        if self.eval_epoch_interest_set and sample_list:
            for sample in sample_list:
                eval_step_out.append(sample.select(
                    self.eval_epoch_interest_set))

        return eval_step_out

    def validation_epoch_end(self, step_out_all):
        # single dataloader case
        if len(step_out_all[0]) == 0 or not isinstance(step_out_all[0][0], list):
            self.validation_epoch_end_one(step_out_all, 0)
        else:  # multiple dataloader case
            for i, step_out in enumerate(step_out_all):
                self.validation_epoch_end_one(step_out, i)

    def validation_epoch_end_one(self, step_output_list, dl_idx):
        # collate epoch_output
        sample_list = list(itertools.chain.from_iterable(step_output_list))

        # eval epoch end callback hook
        if self.eval_epoch_hook is not None:
            if isinstance(self.eval_epoch_hook, list):
                for hook in self.eval_epoch_hook:
                    hook(sample_list, self)
            else:
                self.eval_epoch_hook(sample_list, self)

        # format
        gt_path = None
        pred_path = self.formatted_path
        if self.eval_evaluate:
            _, gt_path = tempfile.mkstemp(suffix='', prefix='mdet_gt_')
            if pred_path is None:
                _, pred_path = tempfile.mkstemp(
                    suffix='', prefix='mdet_pred_')
        pred_path, gt_path = self.eval_datasets[dl_idx].format(
            sample_list, pred_path=pred_path, gt_path=gt_path)

        # evaluation
        if self.eval_evaluate:
            metric = self.eval_datasets[dl_idx].evaluate(pred_path, gt_path)
            self.log_dict(metric)

            os.unlink(gt_path)
            if self.formatted_path is None:
                os.unlink(pred_path)

    def train_dataloader(self):
        dl_cfg = self.config['data']['train']['dataloader']
        return DataLoader(ConcatDataset(self.train_datasets), collate_fn=self.train_collater, **dl_cfg)

    def val_dataloader(self):
        dl_cfg = self.config['data']['eval']['dataloader']
        return [DataLoader(ds, collate_fn=self.eval_collater, **dl_cfg) for ds in self.eval_datasets]

    def export_dataloader(self):
        dl_cfg = self.config['data']['export']
        return DataLoader(self.export_dataset, collate_fn=self.export_collater, **dl_cfg)

    @property
    def train_datasets(self):
        if not hasattr(self,  '_train_datasets'):
            ds_cfg = self.config['data']['train']['dataset']
            if isinstance(ds_cfg, list):
                self._train_datasets = [FI.create(cfg) for cfg in ds_cfg]
            else:
                self._train_datasets = [FI.create(ds_cfg)]
            for ds in self._train_datasets:
                ds.codec = self.train_codec
        return self._train_datasets

    @property
    def train_dataset(self):
        return self.train_datasets[0]

    @property
    def eval_datasets(self):
        if not hasattr(self,  '_eval_datasets'):
            ds_cfg = self.config['data']['eval']['dataset']
            if isinstance(ds_cfg, list):
                self._eval_datasets = [
                    FI.create(cfg) for cfg in ds_cfg]
            else:
                self._eval_datasets = [FI.create(ds_cfg)]
            for ds in self._eval_datasets:
                ds.codec = self.eval_codec
        return self._eval_datasets

    @property
    def eval_dataset(self):
        return self.eval_datasets[0]

    @property
    def export_datasets(self):
        if not hasattr(self,  '_export_datasets'):
            ds_cfg = self.config['data']['export']['dataset']
            if isinstance(ds_cfg, list):
                self._export_datasets = [
                    FI.create(cfg) for cfg in ds_cfg]
            else:
                self._export_datasets = [FI.create(ds_cfg)]
            for ds in self._export_datasets:
                ds.codec = self.export_codec
        return self._export_datasets

    @property
    def export_dataset(self):
        return self.export_datasets[0]

    def configure_optimizers(self):
        # optimizer
        optim_cfg = self.config['fit']['optimizer'].copy()
        optim_type_name = optim_cfg.pop('type')
        optimizer = optim.__dict__[optim_type_name](
            self.train_model.parameters(), **optim_cfg)

        # lr scheduler
        sched_cfg = self.config['fit']['scheduler'].copy()
        sched_type_name = sched_cfg.pop('type')

        # hack lr_scheduler config for specific lr_scheduler
        sched_interval = 'epoch'
        if sched_type_name == 'OneCycleLR':
            sched_cfg['total_steps'] = math.ceil(len(self.train_dataloader(
            ))/self.config['ngpu']) * self.config['fit']['max_epochs']
            sched_interval = 'step'

        scheduler = optim.lr_scheduler.__dict__[
            sched_type_name](optimizer, **sched_cfg)

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval=sched_interval,
            ),
        )

    def on_validation_start(self):
        if self.trt_model is None:  # in case we are in trt eval mode, track is not needed
            self.track_model(self.train_model, self.eval_model)

        # flags for evaluation
        self.evaluate_min_epoch = self.config['runtime']['eval'].get(
            'evaluate_min_epoch', 0)
        self.eval_evaluate_loss = self.config['runtime']['eval'].get(
            'evaluate_loss', True)
        self.eval_log_loss = self.config['runtime']['eval'].get(
            'log_loss', True)
        self.eval_evaluate = self.config['runtime']['eval'].get(
            'evaluate', True) and self.current_epoch >= self.evaluate_min_epoch
        self.eval_interest_set = self.config['runtime']['eval'].get(
            'interest_set', set())
        self.eval_epoch_interest_set = self.config['runtime']['eval'].get(
            'epoch_interest_set', set())
        self.eval_step_hook = self.config['runtime']['eval'].get(
            'step_hook', None)
        self.eval_epoch_hook = self.config['runtime']['eval'].get(
            'epoch_hook', None)
        if self.eval_evaluate:
            self.eval_interest_set |= {'anno', 'pred', 'meta'}
            self.eval_epoch_interest_set |= {'anno', 'pred', 'meta'}
        self.formatted_path = self.config['runtime']['eval'].get(
            'formatted_path', None)

    @ property
    def eval_model(self):
        return self.trt_model or self.util_models['eval_model']

    @ property
    def export_model(self):
        return self.util_models['export_model']

    def track_model(self, m_target, m_follow, name_str=''):
        # track parameters
        m_follow._parameters = m_target._parameters

        # track buffers
        m_follow._buffers = m_target._buffers

        # recursively track submodules
        for name in m_follow._modules:
            self.track_model(
                m_target._modules[name], m_follow._modules[name], f'{name_str}.{name}')

    def export_onnx(self, output_file='tmp.onnx', **kwargs):
        # model
        self.cuda()
        self.track_model(self.train_model, self.export_model)

        # data
        batch = iter(self.export_dataloader()).next().select(
            ['input']).to('cuda')

        input, input_name, output_name, dynamic_axes = self.export_codec.export_info(
            batch)

        # export
        torch.onnx.export(self,
                          input,
                          output_file,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=dynamic_axes,
                          keep_initializers_as_inputs=False,
                          opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                          verbose=True,
                          **kwargs)

    def export_torch(self, output_file='tmp.pt', **kwargs):
        # model
        self.cuda()
        self.track_model(self.train_model, self.export_model)

        # data
        batch = iter(self.export_dataloader()).next().select(
            ['input']).to('cuda')

        input, input_name, output_name, dynamic_axes = self.export_codec.export_info(
            batch)

        traced_module = torch.jit.trace(self, input)

        print(traced_module.graph)

        traced_module.save(output_file)
