# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# A MinGPT + Lightning + xFormers example Code from Sean Naren (@seannaren)
# This is an hommage to https://github.com/karpathy/minGPT

import math
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, TensorDataset

from xformers.factory.model_factory import xFormer, xFormerConfig


class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

    def __init__(
        self,
        vocab_size,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        learning_rate=5e-4,
        n_embd=64,
        block_size=1,
        n_layer=4,
        n_head=4,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        mlp_pdrop=0.1,
        attention="scaled_dot_product",
        hidden_layer_multiplier=4,
        warmup_tokens=20,
        final_tokens=1000,
    ):
        super().__init__()

        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": self.hparams.n_layer,
                "dim_model": self.hparams.n_embd,
                "residual_norm_style": "post",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.block_size,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": self.hparams.n_head,
                    "residual_dropout": self.hparams.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": self.hparams.attention,
                        "dropout": self.hparams.attn_pdrop,
                        "causal": True,
                        "seq_len": self.hparams.block_size,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": self.hparams.mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        config.weight_init = "small"
        self.model = xFormer.from_config(config)

        # decoder head
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size, bias=False)

        self.block_size = self.hparams.block_size
        self.apply(self._init_weights)

        self._tokens_seen = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reset the token counter
        self._tokens_seen = 0

    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self):
        # Create the optimizer and the training schedule:
        # - Handle the per-param weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        # optim_groups = [
        #     {"params": params_decay, "weight_decay": self.hparams.weight_decay},
        #     {"params": params_nodecay, "weight_decay": 0.0},
        # ]
        #
        # # - Start with a warm up, ramp up then cosine
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        # )

        optim_groups = [
            {"params": params_decay, "weight_decay": 0.0},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.RAdam(
            optim_groups, lr=self.hparams.learning_rate
        )

        # def update_lr(*_):
        #     config = self.hparams
        #
        #     if self._tokens_seen < config.warmup_tokens:
        #         # linear warmup
        #         lr_mult = float(self._tokens_seen) / float(max(1, config.warmup_tokens))
        #         lr_mult = max(lr_mult, 1e-2)  # could be that we've not seen any yet
        #     else:
        #         # cosine learning rate decay
        #         progress = float(self._tokens_seen - config.warmup_tokens) / float(
        #             max(1, config.final_tokens - config.warmup_tokens)
        #         )
        #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        #
        #     return lr_mult

        # lr_scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         lr_lambda=[update_lr, update_lr],
        #     ),
        #     "name": "learning_rate",
        #     "interval": "step",  # The unit of the scheduler's step size
        #     "frequency": 1,  # The frequency of the scheduler
        # }
        # return [optimizer], [lr_scheduler]
        return [optimizer], []

    def forward(self, src):
        # predict the next tokens (in latent space)
        prediction = self.model(src)

        # translate the predictions into tokens
        prediction = self.ln_f(prediction)
        logits = self.head(prediction)

        return logits

    def training_step(self, batch, batch_idx):
        src, targets, loss_mask = batch
        # print(src.shape, targets.shape, loss_mask.shape)

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits = self(src)
        # print(logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print(logits.view(-1, logits.size(-1)).shape)
            # print(targets.view(-1).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none").view(
                logits.shape[0], logits.shape[1]
            )
            # print(loss)
            # print(loss.shape)
            # print(loss.mean())
            # loss = (loss * loss_mask).mean()

            # print(loss)

            masked_predictions = (loss_mask * logits.argmax(-1)).long()
            correct_predictions = (masked_predictions == targets).detach()
            n_correct = (correct_predictions * loss_mask).sum(-1)
            n_total = loss_mask.sum(-1)
            accuracies = n_correct / n_total

            # loss = (loss * loss_mask).sum()/loss_mask.sum()
            loss = (loss * loss_mask).mean()
        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                # "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            # step=trainer.global_step,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        src, targets, loss_mask = batch
        # print(src.shape, targets.shape, loss_mask.shape)

        # Update the tokens we've seen (tracked for LR scheduling)
        self._tokens_seen += (src >= 0).numel()

        # same action as inference
        logits = self(src)
        # print(logits.shape)

        # if we are given some desired targets also calculate the loss

        loss = None
        if targets is not None:
            # print(logits.view(-1, logits.size(-1)).shape)
            # print(targets.view(-1).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none").view(
                logits.shape[0], logits.shape[1]
            )
            # print(loss)
            # print(loss.shape)
            # print(loss.mean())
            # loss = (loss * loss_mask).mean()

            masked_predictions = (loss_mask * logits.argmax(-1)).long()
            correct_predictions = (masked_predictions == targets).detach()
            n_correct = (correct_predictions * loss_mask).sum(-1)
            n_total = loss_mask.sum(-1)
            accuracies = n_correct / n_total
            token_acc = n_correct.sum() / n_total.sum()
            seq_acc = accuracies.mean()

            # import ipdb
            # if (self.trainer.current_epoch % 2 == 0) and (batch_idx == 0):
            #     ipdb.set_trace()
            # loss = (loss * loss_mask).sum()/loss_mask.sum()
            loss = (loss * loss_mask).mean()
            # print(loss)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_tok_acc", token_acc, prog_bar=True)
        self.log("val_seq_acc", seq_acc, prog_bar=True)
        self.log("val_seq_lm", n_total.mean(), prog_bar=True)
        self.log("val_seq_lv", n_total.var(), prog_bar=True)
        # self.logger.log_metrics(
        #     {
        #         "val_loss": loss.mean(),
        #         # "learning_rate": self.lr_schedulers().get_last_lr()[0],
        #     },
        #     step=trainer.global_step,
        # )
        return loss


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]


        # Toy Copy Task
        # NOTE: block_size is a multiple of 2.
        dix = 2 * dix[:self.block_size//2]

        # src and target are off by one, we want the model to predict the next word
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # y = torch.tensor(dix[:-1], dtype=torch.long)
        mask = torch.zeros(len(dix) - 1)
        mask[(len(dix)//2 - 1):] = 1
        return x, y, mask

    def to_tokens(self, message, device):
        return torch.tensor([self.stoi[s] for s in message], dtype=torch.long)[
            None, ...
        ].to(device)

    def from_tokens(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])


class CausalCopyDataset(IterableDataset):
    # TODO: does this work with multiple workers?
    # based on https://github.com/idiap/linear-transformer-experiments/blob/master/causal-copy/main.py#L23
    def __init__(self, n_classes, n_maxlen, n_seq):
        # chars = list(set(data))
        # data_size, vocab_size = len(data), len(chars)
        # rank_zero_info("data has %d characters, %d unique." % (data_size, vocab_size))

        # self.stoi = {ch: i for i, ch in enumerate(chars)}
        # self.itos = {i: ch for i, ch in enumerate(chars)}
        # n_maxlen is the length of "seq_to_copy + seq_to_copy": "A B C A B C"
        self.n_maxlen = n_maxlen
        self.n_classes = n_classes
        self.n_min_seq = self.n_maxlen // 4
        # TODO: why the -1?
        self.n_max_seq = (self.n_maxlen - 1) // 2
        self.n_seq = n_seq

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_seq

    def __next__(self):
        # TODO: see where/how padding is done. We have sequences with varied inputs.
        # TODO: add start/end tokens. Although the copy task code doesnt seem to have it?
        # print(self.n_min_seq, self.n_max_seq)
        n_curr_seq = torch.randint(self.n_min_seq, self.n_max_seq, ())
        # n_curr_seq = self.n_max_seq # Fixed length for debugging

        # TODO: why the +1? Because the actual tokens start with 1 and the padded tokens are the only 0s
        #  (but they are ignored in the loss_mask anyone so does that even matter)?
        # 0 is padding, 1 is start, 2 is end
        curr_seq = (torch.rand(n_curr_seq) * self.n_classes).long() + 3
        # TODO: This assumes curr_sequence is at least length 2 long. Maybe add a check earlier?
        curr_seq[0], curr_seq[-1] = 1, 2
        # print(len(curr_seq), curr_seq)
        x = torch.zeros(self.n_maxlen, dtype=torch.long)
        y = torch.zeros(self.n_maxlen, dtype=torch.long)
        loss_mask = torch.zeros(self.n_maxlen)
        x[:n_curr_seq] = curr_seq
        # x[(n_curr_seq + 1):(2 * n_curr_seq + 1)] = curr_seq
        x[n_curr_seq:(2 * n_curr_seq)] = curr_seq
        y[:-1] = x[1:]
        loss_mask[(n_curr_seq - 1):(2 * n_curr_seq - 1)] = 1
        # print(x, y, loss_mask)
        return x, y, loss_mask


class NonCausalCopyDataset(Dataset):
    pass

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()

    # CREDITS: https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
    def top_k_logits(logits, k):
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    for _ in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x[0]  # escape the batch dimension


def gen_copy_sequences(n_classes, n_maxlen, n_seq):
    n_min_seq = n_maxlen // 4
    # TODO: why the -1?
    n_max_seq = (n_maxlen - 1) // 2
    xs, ys, masks = [], [], []
    for _ in range(n_seq):
        # print(self.n_min_seq, self.n_max_seq)
        n_curr_seq = torch.randint(n_min_seq, n_max_seq, ())
        # n_curr_seq = n_max_seq # Fixed length for debugging

        # TODO: why the +1? Because the actual tokens start with 1 and the padded tokens are the only 0s
        #  (but they are ignored in the loss_mask anyone so does that even matter)?
        # 0 is padding, 1 is start, 2 is end
        curr_seq = (torch.rand(n_curr_seq) * n_classes).long() + 3
        # TODO: This assumes curr_sequence is at least length 2 long. Maybe add a check earlier?
        curr_seq[0], curr_seq[-1] = 1, 2
        # print(len(curr_seq), curr_seq)
        x = torch.zeros(n_maxlen, dtype=torch.long)
        y = torch.zeros(n_maxlen, dtype=torch.long)
        loss_mask = torch.zeros(n_maxlen)
        x[:n_curr_seq] = curr_seq
        # x[(n_curr_seq + 1):(2 * n_curr_seq + 1)] = curr_seq
        x[n_curr_seq:(2 * n_curr_seq)] = curr_seq
        y[:-1] = x[1:]
        start_loss_idx = n_curr_seq - 1

        loss_mask[start_loss_idx:start_loss_idx + n_curr_seq] = 1
        # print(x, y, loss_mask)
        xs.append(x), ys.append(y), masks.append(loss_mask)

    return torch.vstack(xs), torch.vstack(ys), torch.vstack(masks)


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """
    def on_after_backward(self, trainer, model):
        model.log("my_model/grad_norm", gradient_norm(model))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


if __name__ == "__main__":
    seed_everything(42)
    REF_BATCH = 512
    BATCH = 512  # adjust depending on the avaiable memory on your machine
    # TODO: how to change the IterableDataset class to work with multiple workers?
    WORKERS = 1
    EPOCHS = 5
    BLOCK = 256
    CLASSES = 10
    WARMUP = 20
    N_SEQ = 100000
    N_LAYER = 4
    FRAC_VAL = 0.2

    if not os.path.exists("input.txt"):
        os.system(
            "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    # text = open("input.txt", "r").read()
    # train_dataset = CharDataset(
    #     text, BLOCK
    # )  # one line of poem is roughly 50 characters
    xs, ys, masks = gen_copy_sequences(n_classes=CLASSES, n_maxlen=BLOCK, n_seq=N_SEQ)
    n = len(xs)
    n_train = int(n * (1 - FRAC_VAL))
    train_xs, val_xs = xs[:n_train], xs[n_train:]
    train_ys, val_ys = ys[:n_train], ys[n_train:]
    train_masks, val_masks = masks[:n_train], masks[n_train:]
    train_dataset = TensorDataset(train_xs, train_ys, train_masks)
    val_dataset = TensorDataset(val_xs, val_ys, val_masks)

    # train_dataset = CausalCopyDataset(
    #     n_classes=CLASSES, n_maxlen=BLOCK, n_seq=N_SEQ
    # )  # one line of poem is roughly 50 characters

    # TODO: is this going to be the same first epoch as the train loader.
    #  Generate the train and val sets before hand as opposed to on the fly?
    # val_dataset = CausalCopyDataset(
    #     n_classes=CLASSES, n_maxlen=BLOCK, n_seq=N_SEQ
    # )  # one line of poem is roughly 50 characters

    #
    # random_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        # sampler=random_sampler,
        batch_size=BATCH,
        num_workers=WORKERS,
        pin_memory=True,
        shuffle=True
    )

    # TODO: shuffling
    val_loader = DataLoader(
        val_dataset,
        # sampler=random_sampler,
        batch_size=BATCH,
        num_workers=WORKERS,
        pin_memory=True,
        shuffle=False
    )

    model = GPT(
        vocab_size=CLASSES + 3,
        block_size=BLOCK,
        attention="scaled_dot_product",
        # attention="linformer",
        # attention="nystrom",
        # attention="favor",
        warmup_tokens=REF_BATCH * WARMUP,
        # TODO: fix
        final_tokens=EPOCHS * len(train_dataset) * BLOCK,
        n_layer=N_LAYER,
    )
    print(model)

    trainer = Trainer(
        gpus=1,
        max_epochs=EPOCHS,
        precision=16,
        gradient_clip_val=1,
        log_every_n_steps=1,
        # terminate_on_nan=True,
        accumulate_grad_batches=REF_BATCH // BATCH,
        # accelerator="cpu",
        check_val_every_n_epoch=1,
        callbacks=[GradNormCallback()]
    )

    trainer.fit(model, train_loader, val_loader)

    # sample from the model
    # context = "Friends of my soul"  # Prime with something

    # x = train_dataset.to_tokens(context, model.device)

    x = torch.tensor([1, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 7, 8, 2], dtype=torch.long)[None, ...]
    x = torch.tensor([1, 3, 4, 5, 7, 9, 6, 7, 8, 2], dtype=torch.long)[None, ...]
    # val_xs_len = (val_xs != 0).sum(-1)
    # # x = val_xs[0:10]
    # # model.eval()
    # # logits = model(x)
    # # preds = logits.argmax(-1)

    # x = val_xs[0:1,:val_xs_len[0] // 2]
    y = sample(model, x, steps=20, temperature=1.0, sample=False, top_k=1)

    # import ipdb
    # ipdb.set_trace()

    # print(train_dataset.from_tokens(y))
    print(y)
