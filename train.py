import os
import torch
import torch.optim as optim
import time
import copy
from tqdm import tqdm

from targets import get_targets
from train_utils import setup_seed, generate_samples, plot_losses


class ModelWrapper:
    # wraps main model and ema model for use in get_targets
    def __init__(self, model, ema_model=None):
        self.model = model
        self.ema_model = ema_model
        self.step = 0

    def call_model(self, x, t, dt_base=None, train=True):
        # forward pass through main model
        self.model.train(train)
        with torch.set_grad_enabled(train):
            output = self.model(x, t, dt_base)
        return output.sample

    def call_model_ema(self, x, t, dt_base=None, train=False):
        # forward pass through ema model, fall back to main model if ema not available
        if self.ema_model is not None:
            self.ema_model.eval()
            with torch.no_grad():
                output = self.ema_model(x, t, dt_base)
            return output.sample
        else:
            return self.call_model(x, t, dt_base, train=False)


def extract_images_from_batch(batch):
    # support both (lr, hr) and (lr, hr, metadata) batch formats from custom MRSI dataloader
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    return batch, batch


def train_shortcut_model(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=64,
    learning_rate=1e-4,
    bootstrap_every=8,
    denoise_timesteps=128,
    weight_decay=0.01,
    num_epochs=50,
    device=None,
    num_workers=4,
    save_dir='checkpoints',
    save_every=5,
    seed=42,
    scheduler_type='cosine',
    warmup_steps=1000,
    sample_every=10,
    lr_min=1e-6,
    grad_clip=1.0,
    ema_decay=0.999
):
    setup_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on device', device)

    # move model to device
    model = model.to(device)

    # ema model via deep copy
    ema_model = None
    if ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        print('ema model created')

    # model wrapper for get_targets
    model_wrapper = ModelWrapper(model, ema_model)

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # scheduler
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(train_dataset) // batch_size,
            eta_min=lr_min
        )
    elif scheduler_type == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # loss tracking
    train_losses      = []
    val_losses        = []
    bootstrap_losses  = []
    flow_losses       = []
    best_val_loss     = float('inf')
    start_time        = time.time()

    # training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss           = 0
        epoch_bootstrap_loss = 0
        epoch_flow_loss      = 0
        num_bootstrap_batches = 0

        pbar = tqdm(train_dataset, desc='Epoch %d/%d' % (epoch, num_epochs))
        for batch_idx, batch in enumerate(pbar):
            lr_images, hr_images = extract_images_from_batch(batch)

            # convert to float and move to device
            lr_images = lr_images.float().to(device)
            hr_images = hr_images.float().to(device)

            # stack hr and lr as 2-channel input [hr, lr]
            model_inputs = torch.cat([hr_images, lr_images], dim=1)

            # generate shortcut targets
            x_t, v_t, t, dt_base, info = get_targets(
                batch_size=model_inputs.shape[0],
                bootstrap_every=bootstrap_every,
                denoise_timesteps=denoise_timesteps,
                train_state=model_wrapper,
                images=model_inputs,
                key=epoch * len(train_dataset) + batch_idx,
                use_ema=False
            )

            # loss calculation (get_targets returns targets for flow-matching loss and shortcut loss)
            v_pred = model(x_t, t, dt_base).sample
            loss   = torch.mean((v_pred - v_t) ** 2)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # ema update
            if ema_model is not None:
                with torch.no_grad():
                    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(ema_decay).add_(model_param.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()

            # track bootstrap and flow losses separately
            bootstrap_size = model_inputs.shape[0] // bootstrap_every
            if bootstrap_size > 0:
                num_bootstrap_batches += 1
                bootstrap_loss = torch.mean((v_pred[:bootstrap_size] - v_t[:bootstrap_size]) ** 2).item()
                flow_loss      = torch.mean((v_pred[bootstrap_size:] - v_t[bootstrap_size:]) ** 2).item()
                epoch_bootstrap_loss += bootstrap_loss
                epoch_flow_loss      += flow_loss
                pbar.set_postfix({
                    'loss': loss.item(),
                    'bootstrap': bootstrap_loss,
                    'flow': flow_loss,
                    'lr': optimizer.param_groups[0]['lr']
                })
            else:
                pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        # epoch train loss
        avg_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_loss)

        if num_bootstrap_batches > 0:
            avg_bootstrap_loss = epoch_bootstrap_loss / num_bootstrap_batches
            avg_flow_loss      = epoch_flow_loss      / num_bootstrap_batches
            bootstrap_losses.append(avg_bootstrap_loss)
            flow_losses.append(avg_flow_loss)

        # validation loop
        val_loss = None
        if val_dataset is not None:
            model.eval()
            val_epoch_loss = 0

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataset):
                    lr_images, hr_images = extract_images_from_batch(batch)

                    # skip empty batches
                    if lr_images.numel() == 0 or hr_images.numel() == 0:
                        print('skipping empty batch at index', batch_idx)
                        continue

                    lr_images = lr_images.float().to(device)
                    hr_images = hr_images.float().to(device)

                    model_inputs = torch.cat([hr_images, lr_images], dim=1)

                    x_t, v_t, t, dt_base, info = get_targets(
                        batch_size=model_inputs.shape[0],
                        bootstrap_every=bootstrap_every,
                        denoise_timesteps=denoise_timesteps,
                        train_state=model_wrapper,
                        images=model_inputs,
                        key=10000 + epoch * len(val_dataset) + batch_idx,
                        use_ema=True
                    )

                    # use ema model for validation if available
                    if ema_model is not None:
                        v_pred = ema_model(x_t, t, dt_base).sample
                    else:
                        v_pred = model(x_t, t, dt_base).sample

                    val_epoch_loss += torch.mean((v_pred - v_t) ** 2).item()

            val_loss = val_epoch_loss / len(val_dataset)
            val_losses.append(val_loss)

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(save_dir, 'best_model.pt'))
                print('saved best model - val loss: %.6f' % best_val_loss)

        # print epoch summary
        time_elapsed = time.time() - start_time
        epoch_info   = 'Epoch %d/%d' % (epoch, num_epochs)
        train_info   = 'train loss: %.6f' % avg_loss
        val_info     = 'val loss: %.6f' % val_loss if val_loss is not None else 'val loss: n/a'
        time_info    = '%.4f sec' % time_elapsed
        print(' - '.join((epoch_info, time_info, train_info, val_info)), flush=True)

        if num_bootstrap_batches > 0:
            print('  bootstrap loss: %.6f - flow loss: %.6f' % (avg_bootstrap_loss, avg_flow_loss))

        # save checkpoint
        if epoch % save_every == 0 or epoch == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'bootstrap_losses': bootstrap_losses,
                'flow_losses': flow_losses,
            }, os.path.join(save_dir, 'checkpoint_epoch_%d.pt' % epoch))
            print('saved checkpoint at epoch %d' % epoch)

        # generate samples
        if epoch % sample_every == 0 or epoch == num_epochs:
            generate_samples(
                model=ema_model if ema_model is not None else model,
                step_counts=[1, 4, 16, 128],
                num_samples=4,
                conditioning_images=lr_images[:4],
                ground_truth_images=hr_images[:4],
                device=device,
                save_path=os.path.join(save_dir, 'samples_epoch_%d.png' % epoch)
            )
            print('generated samples at epoch %d' % epoch)

    # plot loss curves
    plot_losses(
        train_losses=train_losses,
        val_losses=val_losses,
        bootstrap_losses=bootstrap_losses,
        flow_losses=flow_losses,
        save_path=os.path.join(save_dir, 'loss_curves.png')
    )

    return {
        'model': model,
        'ema_model': ema_model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'bootstrap_losses': bootstrap_losses,
        'flow_losses': flow_losses,
        'best_val_loss': best_val_loss
    }