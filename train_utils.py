import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def setup_seed(seed):
    # set all random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_samples(model, step_counts, num_samples=4, conditioning_images=None,
                     ground_truth_images=None, image_size=(64, 64), device=None, save_path=None, seed=42):
    # generate samples at multiple step counts and save as a grid image
    # columns: conditioning, ground truth, step_count_1, step_count_2, etc 
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    torch.manual_seed(seed)

    # use random conditioning if none provided
    if conditioning_images is None:
        conditioning_images = torch.randn(num_samples, 1, *image_size, device=device)
    else:
        conditioning_images = conditioning_images[:num_samples]

    rows = num_samples
    cols = len(step_counts) + 2 
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if rows == 1:
        axes = axes.reshape(1, -1)

    # LR metabolic map conditioning column
    for i in range(num_samples):
        cond_img = conditioning_images[i, 0].cpu().numpy()
        cond_img = (cond_img - cond_img.min()) / (cond_img.max() - cond_img.min() + 1e-8)
        axes[i, 0].imshow(cond_img, cmap='jet')
        axes[i, 0].set_title('Conditioning')
        axes[i, 0].axis('off')

    # ground truth column
    if ground_truth_images is not None:
        ground_truth_images = ground_truth_images[:num_samples]
        for i in range(num_samples):
            gt_img = ground_truth_images[i, 0].cpu().numpy()
            gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-8)
            axes[i, 1].imshow(gt_img, cmap='jet')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

    # sample at each step count
    with torch.no_grad():
        for col_idx, steps in enumerate(step_counts, start=2):
            # start from pure noise and denoise for `steps` steps
            x_target = torch.randn(num_samples, 1, *image_size, device=device)
            x        = torch.cat([x_target, conditioning_images], dim=1)
            delta_t  = 1.0 / steps

            for step in range(steps):
                t       = torch.ones(num_samples, device=device) * step / steps
                dt_base = torch.ones_like(t, dtype=torch.int32) * int(np.log2(steps))
                v_pred  = model(x, t, dt_base).sample

                # update only the target channel, keep conditioning fixed
                x_new         = x.clone()
                x_new[:, 0:1] = x[:, 0:1] + v_pred * delta_t
                x             = x_new

            # mask background using ground truth
            x = torch.mul(x, (ground_truth_images != 0).float())

            for row_idx in range(num_samples):
                sample_img = x[row_idx, 0].cpu().numpy()
                sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-8)
                axes[row_idx, col_idx].imshow(sample_img, cmap='jet')
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title('%d steps' % steps)
                axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_losses(train_losses, val_losses=None, bootstrap_losses=None, flow_losses=None, save_path=None):
    # plot train, val, bootstrap, and flow losses over epochs
    plt.figure(figsize=(12, 8))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')

    if val_losses is not None and len(val_losses) > 0:
        val_epochs = epochs
        # if val was not logged every epoch, reconstruct the x-axis
        if len(val_losses) < len(train_losses):
            val_epochs = list(range(1, len(train_losses) + 1, len(train_losses) // len(val_losses)))[:len(val_losses)]
            if len(val_epochs) < len(val_losses):
                val_epochs = val_epochs + [epochs[-1]]
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')

    if bootstrap_losses is not None and len(bootstrap_losses) > 0:
        plt.plot(epochs, bootstrap_losses, 'g-', label='Bootstrap Loss')

    if flow_losses is not None and len(flow_losses) > 0:
        plt.plot(epochs, flow_losses, 'm-', label='Flow Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()