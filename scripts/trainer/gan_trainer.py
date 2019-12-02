# -*- coding: utf-8 -*-
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from .evaluator import BestModelSaver, SnapshotSaver

# Logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())


class GanTrainer():
    _state_dict = ('train_iter', 'test_iter', 'generator', 'discriminator',
                   'gen_model_saver', 'dis_model_saver', 'epoch')

    loss_keys = ('gen_loss', 'real_loss', 'fake_loss',
                 'dis_loss', 'total_loss')

    def __init__(self, save_dir, train_loader, test_loader,
                 generator, discriminator, gen_optimizer, dis_optimizer,
                 loss_fn, device=None, evaluator=None):
        self.train_iter = iter(train_loader)
        self.test_iter = iter(test_loader)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.device = device
        self.evaluator = evaluator
        self.loss_fn = loss_fn
        self.logger = SummaryWriter(save_dir)
        self.epoch = 0

        # Setup saver
        gen_name = 'best_generator'
        dis_name = 'best_discriminator'
        self.gen_model_saver = BestModelSaver(save_dir, gen_name)
        self.dis_model_saver = BestModelSaver(save_dir, dis_name)
        self.snapshot_saver = SnapshotSaver(save_dir)

    def state_dict(self):
        state = dict()
        for key in self._state_keys:
            obj = getattr(self, key)
            if hasattr(obj, 'state_dict'):
                state[key] = obj.state_dict()
            else:
                state[key] = obj

    def load_state_dict(self, state):
        for lbl in state.keys():
            obj = getattr(self, lbl)
            if hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(state[lbl])
            else:
                setattr(self, lbl, state[lbl])

    def _forward(self, inp, out, backward):
        inp = inp.to(device=self.device)
        real = out.to(device=self.device)

        # Forward generator
        fake = self.generator.forward(inp)
        gen_loss = self.loss_fn(fake)

        # Forward discriminator
        judge_real = self.discriminator.forward(real)
        judge_fake = self.discriminator.forward(fake)
        real_loss = torch.abs(1.0 - judge_real)
        fake_loss = torch.abs(judge_fake)
        dis_loss = torch.max(real_loss, fake_loss)

        # Apply discriminator loss
        total_loss = np.mean(gen_loss, 1.0 - fake_loss.detach())

        if backward:
            # Backward generator
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()
            # Backward discriminator
            self.dis_optimizer.zero_grad()
            dis_optimizer.backward()
            self.dis_optimizer.step()

        # Loss
        loss = {
            "gen_loss": gen_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item(),
            'dis_loss': dis_loss.item(),
            'total_loss': total_loss.item(),
        }

        return (fake, loss)

    def _train_step(self, n_train):
        self.generator.train()
        self.discriminator.train()

        avg_loss = {k:0 for k in self.loss_keys}

        for _ in tqdm(range(n_train)):
            data = next(self.train_iter)
            _, loss = self._forward(data['input'], data['output'], True)
            for k, v in loss.items():
                avg_loss[k] += v

        for k, v in avg_loss.items():
            self.logger.add_scalar('train_%s' % k, v / n_train, self.epoch)
            logger.info('train_%s: %f', k, v)

    def _test_step(self):
        self.generator.eval()
        self.discriminator.eval()

        n_test = len(self.test_iter.loader.dataset)

        avg_loss = {k:0 for k in self.loss_keys}

        for _ in tqdm(range(n_test)):
            with torch.no_grad():
                data = next(self.train_iter)
                pred, loss = self._forward(data['input'], data['output'],
                                           False)
                for k, v in loss.items():
                    avg_loss[k] += v
                if self.evaluator is not None:
                    self.evaluator.evaluate(data, pred, self.epoch)

        for k, v in avg_loss.items():
            self.logger.add_scalar('test_%s' % k, v / n_train, self.epoch)
            logger.info('test_%s: %f', k, v)

        self.gen_model_saver(avg_loss['gen_loss'], self.generator)
        self.dis_model_saver(avg_loss['dis_loss'], self.discriminator)

    def run(self, n_train, max_epoch=-1):
        while True:
            self.epoch += 1
            logger.info('Epoch: %d', self.epoch)
            self._train_step(n_train)
            self._test_step()

            if 0 < max_epoch < self.epoch:
                logger.info('Reached max epoch')
                break
