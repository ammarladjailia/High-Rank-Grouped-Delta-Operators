import torch
import os

def save_checkpoint(model, optimizer, scheduler, seed, epoch, config, val_accuracy, path):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'seed': seed,
        'epoch': epoch,
        'config': config,
        'val_accuracy': val_accuracy
    }, path)
