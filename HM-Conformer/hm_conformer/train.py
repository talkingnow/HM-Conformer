from tqdm import tqdm

import torch
import torch.distributed as dist

from egg_exp.util import df_test
import egg_exp.util as util

def train(epoch, framework, optimizer, loader, logger):
    framework.train()
    
    count = 0
    loss_sum = 0
    loss_sum_list = [0]*5

    with tqdm(total=len(loader), ncols=90) as pbar:
        for x, label in loader:
            # to GPU
            x = x.to(torch.float32).to(framework.device)
            label = label.to(framework.device)
            
            # clear grad
            optimizer.zero_grad()
            
            # feed forward
            _, loss, loss_embs = framework(x, label)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            
            # logging
            if logger is not None:
                count += 1
                loss_sum += loss.item()
                for i in range(5):
                    loss_sum_list[i] += loss_embs[i].item()

                if len(loader) * 0.02 <= count:
                    logger.log_metric('Loss', loss_sum / count)
                    loss_sum = 0
                    for i in range(5):
                        logger.log_metric(f'Loss{i}', loss_sum_list[i] / count)
                        loss_sum_list[i] = 0
                    count = 0

                desc = f'[{epoch}|(loss): {loss.item():.4f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def test(framework, loader):
    # enrollment
    eer = df_test(framework, loader, run_on_ddp=True, get_scores=False)
    return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()