# -*- coding: utf-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
from .evaluator import BestModelSaver, SnapshotSaver


class CycleGanTrainer():
    _state_keys = ('train_iter', 'test_iter', 'a2b', 'b2a', 'dis_a', 'dis_b',
                   'a2b_gen_saver', 'b2a_gen_saver', 'a_dis_saver',
                   'b_dis_saver', 'epoch')
    loss_keys = ('gen_a2b_gan_loss', 'gen_b2a_gan_loss', 'gen_a_identify_loss',
                 'gen_b_identify_loss', 'gen_a2b2a_cycle_loss',
                 'gen_b2a2b_cycle_loss', 'gen_loss', 'dis_a_real_loss',
                 'dis_b_real_loss', 'dis_a_loss', 'dis_b_loss')

    def __init__(self, train_loader, test_loader, gen_a2b, gen_b2a,
                 dis_a, dis_b, gen_optimizer, dis_optimizer,
                 label_a=None, lebel_b=None, device=None, evaluator=None):
        self.train_iter = iter(train_loader)
        self.test_iter = iter(test_loader)
        self.a2b = gen_a2b
        self.b2a = gen_b2a
        self.dis_a = dis_a
        self.dis_b = dis_b
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.lebel_a = 'A' if label_a is None else label_a
        self.lebel_b = 'B' if lbbel_b is None else lbbel_b
        self.device = device
        self.evaluator = evaluator
        self.logger = SummaryWriter(save_dir)
        self.epoch = 0

        # Setup saver
        a2b_name = 'best_%s_to_%s' % (self.label_a, self.label_b)
        b2a_name = 'best_%s_to_%s' % (self.label_b, self.label_a)
        dis_name = 'best_discriminator_%s'
        self.a2b_gen_saver = BestModelSaver(save_dir, a2b_name)
        self.b2a_gen_saver = BestModelSaver(save_dir, b2a_name)
        self.a_dis_saver = BestModelSaver(save_dir, dis_name % 'A')
        self.b_dis_saver = BestModelSaver(save_dir, dis_name % 'B')
        self.snapshot_saver = SnapshotSaver(save_dir)

        # Loss function
        self.idt_loss_fn = nn.L1Loss()
        self.cycle_loss_fn = nn.L1Loss()

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

    def _forward(self, inp, backward):
        a = inp['a'].to(self.device)
        b = inp['b'].to(self.device)

        # Forward Generator
        ab = self.b2a(inp_a)
        ba = self.a2b(inp_b)
        aba = self.b2a(ab)
        bab = self.a2b(ba)
        aa = self.b2a(inp_a)
        bb = self.a2b(inp_b)

        # Disable discriminator loss
        self.dis_a.requires_grad_(False)
        self.dis_b.requires_grad_(False)

        # Compute Loss
        ab_loss = self.dis_b(ab)
        ba_loss = self.dis_a(ba)
        idt_a_loss = self.idt_loss_fn(aa, inp_a)
        idt_b_loss = self.idt_loss_fn(bb, inp_b)
        cycle_a_loss = self.cycle_loss_fn(aba, inp_a)
        cycle_b_loss = self.cycle_loss_fn(bab, inp_b)
        gen_loss = torch.mean(ab_loss, ba_loss, idt_a_loss, idt_b_loss,
                              cycle_a_loss, cycle_b_loss)

        # Backward generator
        if backward:
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

        # enable discriminator loss
        self.dis_a.requires_grad_(True)
        self.dis_b.requires_grad_(True)

        # Forward discriminator
        real_a_loss = 1.0 - self.dis_a(a)
        real_b_loss = 1.0 - self.dis_b(b)

        # Backward discriminator
        if backward:
            ab_loss = self.dis_b(ab.detach())
            ba_loss = self.dis_a(ba.detach())
            self.dis_optimizer.zero_grad()
            dis_a_loss = torch.max(real_a_loss, ba_loss)
            dis_b_loss = torch.max(real_b_loss, ab_loss)
            dis_a_loss.backward()
            dis_b_loss.backward()
            self.dis_optimizer.step()
        else:
            dis_a_loss = torch.max(real_a_loss, ba_loss)
            dis_b_loss = torch.max(real_b_loss, ab_loss)

        # Loss
        loss = {
            'gen_a2b_gan_loss': ab_loss.item(),
            'gen_b2a_gan_loss': ba_loss.item(),
            'gen_a_identify_loss': idt_a_loss.item(),
            'gen_b_identify_loss': idt_b_loss.item(),
            'gen_a2b2a_cycle_loss': cycle_a_loss.item(),
            'gen_b2a2b_cycle_loss': cycle_b_loss.item(),
            'gen_loss': gen_loss.item(),
            'dis_a_real_loss': real_a_loss.item(),
            'dis_b_real_loss': real_b_loss.item(),
            'dis_a_loss': dis_a_loss.item(),
            'dis_b_loss': dis_b_loss.item(),
        }

        # Predoction result
        pred = dict(a=ab, b=ba)

        return (pred, loss)

    def _train_step(self, n_train):
        self.a2b.train()
        self.b2a.train()
        self.dis_a.train()
        self.dis_b.train()

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
        self.a2b.eval()
        self.b2a.eval()
        self.dis_a.eval()
        self.dis_b.eval()

        n_test = len(self.test_iter.loader.dataset)

        avg_loss = {k:0 for k in self.loss_keys}

        for _ in tqdm(range(n_test)):
            data = next(self.test_iter)
            pred, loss = self._forward(data['input'], data['output'], False)
            for k, v in loss.items():
                avg_loss[k] += v
            if self.evaluator is not None:
                self.evaluator.evaluate(self.epoch, data, pred)

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
