# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

def get_trainer(args):
    from infinity.trainer.sft_trainer import InfinityTrainer as Trainer
    return Trainer