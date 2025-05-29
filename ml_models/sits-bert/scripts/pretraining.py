import argparse
import random

import numpy as np
import torch

from dataset.dataset_wrapper import DataSetWrapper
from sitsbert.model import SBERT
from trainer import SBERTTrainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set random seed.
setup_seed(123)


def pre_training_configuration():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=False,
        help="The input pre training data path.",
    )
    parser.add_argument(
        "--pretrain_path",
        default=None,
        type=str,
        required=False,
        help="The output directory where the pre-training checkpoints will be "
        "written.",
    )
    parser.add_argument(
        "--valid_rate", 
        default=0.03, 
        type=float, 
        help="Proportion of training data to use for validation in "
        "pre-training stage. This should be a float between 0.0 and 1.0, "
        "where 0.0 means no validation set and 1.0 means all data is used for "
        "validation.",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="The maximum length of input time series. Context size. Sequences "
        "longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_features",
        default=10,
        type=int,
        help="The dimensionality of satellite observations. Default is 10 bands "
        "for sentinel 2 surface reflectance.",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs for pre-training. This is the number of times "
        "the model will see the entire training dataset during pre-training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="Hidden size of the SITS-BERT model.",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="Number of transformer blocks (layers).",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="Number of attention heads in each transformer block.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--decay_gamma",
        default=0.99,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--gradient_clipping",
        default=5.0,
        type=float,
        help="",
    )

    return parser.parse_args()


if __name__ == "__main__":
    
    
    pre_training_configuration = pre_training_configuration()

    print(">>> Loading training and validation data sets ...")
    
    # Interface for creating PyTorch DataLoaders for training and validation datasets.
    dataset = DataSetWrapper(
        batch_size=pre_training_configuration.batch_size,
        valid_size=pre_training_configuration.valid_rate,
        data_path=pre_training_configuration.dataset_path,
        num_features=pre_training_configuration.num_features,
        max_length=pre_training_configuration.max_length,
    )

    # Get the training and validation DataLoaders.
    train_loader, valid_loader = dataset.get_data_loaders()

    print(">>> Initialing SITS-BERT ...")
    

    sbert = SBERT(
        pre_training_configuration.num_features,
        hidden=pre_training_configuration.hidden_size,
        n_layers=pre_training_configuration.layers,
        attn_heads=pre_training_configuration.attn_heads,
        dropout=pre_training_configuration.dropout,
    )

    trainer = SBERTTrainer(
        sbert,
        pre_training_configuration.num_features,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        lr=pre_training_configuration.learning_rate,
        warmup_epochs=pre_training_configuration.warmup_epochs,
        decay_gamma=pre_training_configuration.decay_gamma,
        gradient_clipping_value=pre_training_configuration.gradient_clipping,
    )

    print("Pre-training SITS-BERT...")
    mini_loss = np.Inf
    for epoch in range(pre_training_configuration.epochs):
        train_loss, valida_loss = trainer.train(epoch)
        if mini_loss > valida_loss:
            mini_loss = valida_loss
            trainer.save(epoch, pre_training_configuration.pretrain_path)
