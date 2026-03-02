# --------------------------------------------------------
# References:
#    mae: https://github.com/facebookresearch/mae/blob/main/util/misc.py
# --------------------------------------------------------
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import torch
import torch.distributed as dist

import argparse
import inspect
import types

torch_version = torch.__version__
print(torch_version)

major_version, minor_version = map(int, torch_version.split('.')[:2])
# Repair this issue on different version https://github.com/pytorch/pytorch/pull/94709
print(f"Current Torch version: {major_version}.{minor_version}")
if major_version >= 2:
    from torch import inf
else:
    from torch._six import inf

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        # self.total += value * n
        self.total += value # No n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0] + 0.5) # round(), otherwise may return a number minus 1
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, _n=1, **kwargs):
        if len(kwargs) == 0:
            raise "MetricLogger error, no keyword passed"
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=_n)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if len(iterable) == 0:
            print('Total time: {} (Alert! No Iteration in this sample!)'.format(
                total_time_str, total_time))
        else:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")
        self._nan_skip_count = 0
        self._last_nan_skipped = False

    def _sync_nan_flag(self, has_nan: bool) -> bool:
        """
        Synchronize NaN detection across all ranks in distributed training.
        Returns True if ANY rank detected NaN, ensuring all ranks take the same code path.
        This prevents NCCL AllReduce desynchronization when one rank skips a batch.
        """
        if not is_dist_avail_and_initialized():
            return has_nan

        # Create a tensor with 1.0 if NaN detected, 0.0 otherwise
        nan_flag = torch.tensor([1.0 if has_nan else 0.0], device='cuda')
        # Use MAX reduction: if any rank has NaN (1.0), result will be 1.0
        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
        result = nan_flag.item() > 0.5
        # Explicitly free the tensor to avoid memory accumulation
        del nan_flag
        return result

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._last_nan_skipped = False
        # Check if loss is NaN or Inf before backward
        local_nan_detected = not torch.isfinite(loss).all()

        # Synchronize NaN detection across all ranks to prevent NCCL desync
        any_rank_has_nan = self._sync_nan_flag(local_nan_detected)

        if any_rank_has_nan:
            if local_nan_detected:
                loss_val = loss.item() if loss.numel() == 1 else 'tensor'
                print(f"[Warning] Loss is NaN/Inf (value={loss_val}), skipping batch")
            else:
                print(f"[Warning] Another rank detected NaN/Inf loss, skipping batch to stay synchronized")
            self._nan_skip_count += 1
            self._last_nan_skipped = True
            # Run backward even on NaN loss to free the computational graph's
            # saved tensors. This is the only reliable way to release the
            # intermediate activations from the forward pass. The resulting
            # NaN gradients are immediately cleared by zero_grad.
            try:
                loss.backward()
            except RuntimeError:
                pass
            optimizer.zero_grad(set_to_none=True)
            del loss
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device='cpu')

        # Try backward pass with NaN detection
        backward_nan_detected = False
        try:
            self._scaler.scale(loss).backward(create_graph=create_graph)
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                backward_nan_detected = True
                backward_error_msg = str(e)
            else:
                raise  # Re-raise if it's a different error

        # Synchronize backward NaN detection across all ranks
        any_rank_backward_nan = self._sync_nan_flag(backward_nan_detected)

        if any_rank_backward_nan:
            if backward_nan_detected:
                print(f"[Warning] NaN/Inf detected during backward pass, skipping batch: {backward_error_msg}")
            else:
                print(f"[Warning] Another rank detected NaN/Inf during backward, skipping batch to stay synchronized")
            self._nan_skip_count += 1
            self._last_nan_skipped = True
            optimizer.zero_grad()
            # Update scaler to reduce scale factor
            self._scaler.update()
            # Free memory
            del loss
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device='cpu')

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            # Check for NaN/Inf in gradients after unscale
            local_grad_nan = not torch.isfinite(norm)

            # Synchronize gradient NaN detection across all ranks
            any_rank_grad_nan = self._sync_nan_flag(local_grad_nan)

            if any_rank_grad_nan:
                if local_grad_nan:
                    print(f"[Warning] NaN/Inf gradients detected after unscale (norm={norm}), skipping optimizer step")
                else:
                    print(f"[Warning] Another rank detected NaN/Inf gradients, skipping optimizer step to stay synchronized")
                self._nan_skip_count += 1
                self._last_nan_skipped = True
                optimizer.zero_grad()
                self._scaler.update()
                # Free memory
                torch.cuda.empty_cache()
                return torch.tensor(0.0, device='cpu')

            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    @property
    def nan_skip_count(self):
        """Returns the number of batches skipped due to NaN/Inf."""
        return self._nan_skip_count

    @property
    def last_nan_skipped(self):
        """Returns True if the last call to __call__ resulted in a NaN/Inf skip."""
        return self._last_nan_skipped


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    
def all_reduce(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        return x_reduce.item()
    else:
        return x
    
    
import numpy as np
def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



def create_argparser(model_class):
    parser = argparse.ArgumentParser(description=f"Arguments for {model_class.__name__}")
    
    # 获取模型的__init__方法的签名
    if isinstance(model_class,(types.FunctionType, types.MethodType)):
        sig = inspect.signature(model_class)
    else:
        sig = inspect.signature(model_class.__init__)
    # 解析每个参数并添加到argparse
    for name, param in sig.parameters.items():
        if name == 'self' :
            continue
        if name == 'args' or name == 'kwargs':
            raise KeyError('args and kwargs are not supported when initialize model.')
        arg_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        default_value = param.default if param.default != inspect.Parameter.empty else None
        if default_value is not None:
            parser.add_argument(f'--{name}', type=arg_type, default=default_value, help=f'{name} (default: {default_value})')
        else:
            parser.add_argument(f'--{name}', type=arg_type, help=f'{name}')

    return parser