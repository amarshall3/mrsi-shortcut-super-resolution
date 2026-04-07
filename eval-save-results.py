import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse
import matplotlib.pyplot as plt

from shortcut_model import UNet2DModel
from data_loader import create_data_loaders
from train_utils import setup_seed

try:
    import lpips
    _has_lpips = True
except Exception:
    _has_lpips = False


def _compute_lpips(img, gt, loss):
    # compute lpips between ground truth metabolic map and predicted metabolic map
    device = next(loss.parameters()).device
    img_t = torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt ).float().to(device).unsqueeze(0).unsqueeze(0)

    # resize to 64x64 if needed
    if img_t.shape[-2:] != (64, 64):
        img_t = torch.nn.functional.interpolate(img_t, size=(64, 64), mode='bilinear', align_corners=True)
        gt_t = torch.nn.functional.interpolate(gt_t,  size=(64, 64), mode='bilinear', align_corners=True)

    # expand to 3 channels (required for LPIPS)
    img_t = img_t.expand(1, 3, 64, 64)
    gt_t = gt_t.expand(1, 3, 64, 64)

    # normalize to [-1, 1]
    def _norm(x):
        xmax = torch.amax(x)
        return (x * 2 - xmax) / (xmax + 1e-8)

    img_t = _norm(img_t)
    gt_t = _norm(gt_t)

    with torch.no_grad():
        val = loss(img_t, gt_t).detach().cpu().numpy().squeeze()
    return float(val)


def calculate_metrics(pred, target):
    # function for calculating NRMSE, PSNR, and SSIM between predicted and target metabolic maps
    pmin, pmax = pred.min(), pred.max()
    tmin, tmax = target.min(), target.max()
    pred_n = (pred   - pmin) / (pmax - pmin + 1e-8)
    targ_n = (target - tmin) / (tmax - tmin + 1e-8)

    nrmse_val = normalized_root_mse(targ_n, pred_n, normalization='euclidean')
    psnr_val = psnr(targ_n, pred_n, data_range=1.0)
    ssim_val = ssim(targ_n, pred_n, data_range=1.0)

    return {'nrmse': float(nrmse_val), 'psnr': float(psnr_val), 'ssim': float(ssim_val)}


def save_panel_png(out_path, lr_img, hr_img, pred_img, diff_gain=3.0, vmin=None, vmax=None):
    # save side-by-side panel: LR, HR, Output, Diff with jet colormap

    def _tile3(x):
        # upsample for display
        return np.repeat(np.repeat(x, 3, axis=0), 3, axis=1)

    lr_plot = _tile3(lr_img)
    hr_plot = _tile3(hr_img)
    out_plot = _tile3(pred_img)
    diff = np.abs(out_plot - hr_plot) * float(diff_gain)

    strip = np.concatenate([lr_plot, hr_plot, out_plot, diff], axis=1)

    # mask background and low PSNR regions
    single_mask = _tile3(hr_img != 0)
    mask = np.tile(single_mask, (1, 4))
    masked_strip = np.ma.masked_where(mask == 0, strip)

    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('black')  # set background to black 

    if vmin is None: vmin = hr_plot.min()
    if vmax is None: vmax = hr_plot.max()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
    ax.imshow(masked_strip, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, facecolor='black', edgecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def generate_and_evaluate(model, test_loader, step_counts, device, root_save_dir):
    # function to run sampling with different step counts, compute metrics, and save results
    
    model.eval()

    # lpips metric initialization
    lpips_alex = None
    lpips_vgg = None
    if _has_lpips:
        try:
            lpips_alex = lpips.LPIPS(net='alex').to(device).eval()
            lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except Exception:
            lpips_alex = None
            lpips_vgg = None

    results = {}

    for steps in step_counts:
        step_tag = 'steps%d' % steps
        log_dir = os.path.join(root_save_dir, 'logs_%s'   % step_tag)
        img_dir = os.path.join(root_save_dir, 'images_%s' % step_tag)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        run_nrmse = []
        run_psnr = []
        run_ssim = []
        run_lpips = []
        run_lpips_vgg = []

        print('sampling with %d steps' % steps)
        delta_t = 1.0 / steps

        for batch in tqdm(test_loader, desc='steps=%d' % steps):
            # lr metabolic map, hr metabolic map, max value for rescaling, patient id, slice id, metabolite name
            lr_images, met_HR, met_max, patient_str, sli, metname = batch
            hr_images = met_HR.to(device)
            lr_images = lr_images.to(device) # lr_images have same dimension as hr_images (missing k-space information)
            bs = hr_images.shape[0]

            def _to_list(x, B):
                if torch.is_tensor(x):
                    x = x.detach().cpu().tolist()
                if isinstance(x, (str, int, float)):
                    return [x] * B
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                return [str(xx) for xx in x]

            met_max_list = _to_list(met_max,     bs)
            patients =_to_list(patient_str, bs)
            slices = _to_list(sli,         bs)
            metnames = _to_list(metname,     bs)

            # denoise from pure noise using euler's method
            with torch.no_grad():
                x_target = torch.randn(bs, 1, hr_images.shape[2], hr_images.shape[3], device=device)
                x = torch.cat([x_target, lr_images], dim=1)

                for s in range(steps):
                    t = torch.ones(bs, device=device) * (s / steps) # start denoising from t=0 (gaussian noise)
                    dt_base = torch.ones_like(t, dtype=torch.int32) * int(np.log2(steps))
                    v_pred = model(x, t, dt_base).sample
                    x_new = x.clone()
                    x_new[:, 0:1] = x[:, 0:1] + v_pred * delta_t
                    x = x_new

                x = x * (hr_images != 0).float()

            # per-sample metrics and saving
            for i in range(bs):
                mm = float(met_max_list[i])

                # rescale to original physical units before computing metrics
                lr_i = lr_images[i, 0].detach().cpu().numpy() * mm
                hr_i = hr_images[i, 0].detach().cpu().numpy() * mm
                out_i = x[i, 0].detach().cpu().numpy() * mm

                m = calculate_metrics(out_i, hr_i)
                lp_alex= np.nan
                lp_vgg = np.nan
                if _has_lpips and lpips_alex is not None:
                    lp_alex = _compute_lpips(out_i, hr_i, lpips_alex)
                if _has_lpips and lpips_vgg is not None:
                    lp_vgg  = _compute_lpips(out_i, hr_i, lpips_vgg)

                run_nrmse.append(m['nrmse'])
                run_psnr.append(m['psnr'])
                run_ssim.append(m['ssim'])
                if not np.isnan(lp_alex): run_lpips.append(lp_alex)
                if not np.isnan(lp_vgg):  run_lpips_vgg.append(lp_vgg)

                print('%s slice=%s met=%s PSNR=%.4f SSIM=%.4f LPIPS=%.4f' % (
                    patients[i], slices[i], metnames[i],
                    m['psnr'], m['ssim'],
                    0.0 if np.isnan(lp_alex) else lp_alex
                ))

                # save panel png and npz
                png_name = '%s_slice%s_%s_NRMSE%.3g_PSNR%.3g_SSIM%.3g_LPIPS%.3g_LPIPSVGG%.3g.png' % (
                    patients[i], slices[i], metnames[i],
                    m['nrmse'], m['psnr'], m['ssim'],
                    0.0 if np.isnan(lp_alex) else lp_alex,
                    0.0 if np.isnan(lp_vgg)  else lp_vgg
                )
                save_panel_png(os.path.join(img_dir, png_name), lr_i, hr_i, out_i, diff_gain=3.0)

                npz_name = '%s_slice%s_%s.npz' % (patients[i], slices[i], metnames[i])
                np.savez(os.path.join(img_dir, npz_name), met_LR=lr_i, output=out_i, met_HR=hr_i)

        # summarize metrics
        run_nrmse = np.asarray(run_nrmse,     dtype=np.float64)
        run_psnr = np.asarray(run_psnr,      dtype=np.float64)
        run_ssim = np.asarray(run_ssim,      dtype=np.float64)
        run_lp = np.asarray(run_lpips,     dtype=np.float64) if len(run_lpips)     else np.array([np.nan])
        run_lp_v = np.asarray(run_lpips_vgg, dtype=np.float64) if len(run_lpips_vgg) else np.array([np.nan])

        print('NRMSE = %.5g +- %.5g' % (run_nrmse.mean(), run_nrmse.std()))
        print('PSNR  = %.5g +- %.5g' % (run_psnr.mean(),  run_psnr.std()))
        print('SSIM  = %.5g +- %.5g' % (run_ssim.mean(),  run_ssim.std()))
        print('LPIPS = %.5g +- %.5g' % (np.nanmean(run_lp),   np.nanstd(run_lp)))
        print('LPIPS VGG = %.5g +- %.5g' % (np.nanmean(run_lp_v), np.nanstd(run_lp_v)))

        # save metrics arrays
        np.savez(
            os.path.join(root_save_dir, 'metrics_%s.npz' % step_tag),
            nrmse=run_nrmse, psnr=run_psnr, ssim=run_ssim,
            lpips=run_lp, lpips_vgg=run_lp_v
        )

        results[steps] = dict(
            nrmse=float(run_nrmse.mean()), psnr=float(run_psnr.mean()),
            ssim=float(run_ssim.mean()),
            lpips=float(np.nanmean(run_lp)), lpips_vgg=float(np.nanmean(run_lp_v))
        )

    return results


parser = argparse.ArgumentParser()

# model and data
parser.add_argument('--checkpoint',      type=str, default='/gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-final/batch64-bootstrap4-highlr-changeclamp/checkpoint_epoch_200.pt')
parser.add_argument('--data_path',       type=str, default='/gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-main/data_processed')
parser.add_argument('--train_patients',  type=str, default='14,15,16,17,18,19,20,22,79,74,78,81,84,86,91,93,96,98,100')
parser.add_argument('--valid_patients',  type=str, default='8,9,11,12')
parser.add_argument('--test_patients',   type=str, default='1,2,6,8')

# eval
parser.add_argument('--batch_size',  type=int, default=8)
parser.add_argument('--step_counts', type=str, default='1,4,16,32,64,128')
parser.add_argument('--num_workers', type=int, default=4)

# output
parser.add_argument('--output_dir', type=str, default='eval_results')

# hardware
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed',   type=int, default=42)

args = parser.parse_args()

# setup
setup_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
print('running on device', device)

# parse step counts and patient lists
step_counts = [int(x) for x in args.step_counts.split(',')]
train_patients = [int(x) for x in args.train_patients.split(',')]
valid_patients = [int(x) for x in args.valid_patients.split(',')]
test_patients = [int(x) for x in args.test_patients.split(',')]

print('step counts:', step_counts)
print('test patients:', test_patients)

# load checkpoint
print('loading checkpoint from', args.checkpoint)
checkpoint = torch.load(args.checkpoint, map_location=device)

# restore model from checkpoint
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    in_channels = state_dict['conv_in.weight'].shape[1]
    out_channels = state_dict['conv_out.weight'].shape[0]

    # infer block_out_channels from state dict
    block_out_channels = []
    for k in state_dict.keys():
        if 'down_blocks.0.resnets.0.conv1.weight' in k:
            block_out_channels.append(state_dict[k].shape[0])
            break
    for i in range(1, 10):
        k = 'down_blocks.%d.resnets.0.conv1.weight' % i
        if k in state_dict:
            block_out_channels.append(state_dict[k].shape[0])

    print('creating model - in=%d out=%d blocks=%s' % (in_channels, out_channels, block_out_channels))

    model = UNet2DModel(
        in_channels=in_channels,            # noisy target channel + lr conditioning channel
        out_channels=out_channels,          # predicted velocity field
        block_out_channels=tuple(block_out_channels),  # feature channels per unet stage
        layers_per_block=2,                 # resnet layers per block
        handle_delta_time=True,             # embed dt for shortcut step size conditioning
        time_embedding_type='positional',   # sinusoidal time embedding
        attention_head_dim=8,               # attention heads in attn blocks
        norm_num_groups=8,                  # groups for group norm
        dropout=0.1,
        norm_eps=1e-5,
        add_attention=True,                 # add self-attention in mid block
        sample_size=64,                     # spatial resolution
        down_block_types=(
            'DownBlock2D',                  # regular resnet downsampling
            'DownBlock2D',
            'AttnDownBlock2D',              # resnet downsampling with spatial self-attention
            'AttnDownBlock2D',
        ),
        up_block_types=(
            'AttnUpBlock2D',                # resnet upsampling with spatial self-attention
            'AttnUpBlock2D',
            'UpBlock2D',                    # regular resnet upsampling
            'UpBlock2D',
        ),
    )

    # prefer ema weights if available
    if 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
        print('using ema weights')
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        model.load_state_dict(state_dict)
else:
    print('loading full model from checkpoint')
    model = checkpoint

model = model.to(device).eval()

# data loaders
print('creating data loaders...')
_, _, test_loader = create_data_loaders(
    data_path=args.data_path,
    train_patients=train_patients,
    valid_patients=valid_patients,
    test_patients=test_patients,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

# run evaluation
print('starting evaluation...')
_ = generate_and_evaluate(
    model=model,
    test_loader=test_loader,
    step_counts=step_counts,
    device=device,
    root_save_dir=args.output_dir
)

print('done - results saved to', args.output_dir)
