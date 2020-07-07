import os
import argparse

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
import numpy as np
import tqdm
import model


def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    transformer = model.Transformer(
        max_length=max_length,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        sos=sos_idx,
        eos=eos_idx,
        pad=pad_idx)

    file_name = "model/checkpoint.t7"
    if os.path.exists(file_name):
        print("Loading model from file ", file_name)
        transformer.load(file_name)
        print("Loaded.")

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        for epoch in range(args.epochs):
            print("Epoch ", epoch)

            train_losses = []
            val_losses = []

            # training
            pbar = tqdm.tqdm(train_loader, total=227)
            for src_batch, tgt_batch in pbar:
                loss = transformer.train_step(src_batch, tgt_batch)
                pbar.set_description("Loss = {:.6f}".format(loss))
                train_losses.append(loss)
            print("Train loss ", np.mean(train_losses))

            # validation
            for src_batch, tgt_batch in tqdm.tqdm(valid_loader):
                loss = transformer.loss(src_batch, tgt_batch).item()
                val_losses.append(loss)
            print("Validation loss ", np.mean(val_losses))

            print("Saving to ", file_name)
            transformer.save(file_name)
            print("Saved.")
    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        for src_batch, tgt_batch in test_loader:
            pred_batch = transformer.predict(src_batch)

            # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
            # every <pad> token (index: 2) should be located after <eos> token (index: 1).
            # example of pred_batch:
            # [[0, 5, 6, 7, 1],
            #  [0, 4, 9, 1, 2],
            #  [0, 6, 1, 2, 2]]
            pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
