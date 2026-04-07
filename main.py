import argparse
import torch
from pathlib import Path

from shortcut_model import UNet2DModel
from data_loader import create_data_loaders
from train import train_shortcut_model
from train_utils import setup_seed

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--data_path', type=str, default='/gpfs/gibbs/pi/duncan/am3968/MRSI_Project/shortcut-mrsi-main/data_processed')
parser.add_argument('--train_patients', type=str, default='14, 15, 16, 17, 18, 19, 20, 22, 79, 74, 78, 81, 84, 86, 91, 93, 96, 98, 100')
parser.add_argument('--valid_patients', type=str, default='8, 9, 11, 12')
parser.add_argument('--test_patients',  type=str, default='1, 2, 4, 6')

# model parameters
parser.add_argument('--input_channels',     type=int,   default=2)
parser.add_argument('--output_channels',    type=int,   default=1)
parser.add_argument('--block_out_channels', type=str,   default='64,128,192,256')
parser.add_argument('--layers_per_block',   type=int,   default=2)

# training parameters
parser.add_argument('--batch_size',        type=int,   default=32)
parser.add_argument('--num_epochs',        type=int,   default=200)
parser.add_argument('--learning_rate',     type=float, default=1e-4)
parser.add_argument('--weight_decay',      type=float, default=0.01)
parser.add_argument('--bootstrap_every',   type=int,   default=4) # ex: (1/4) of batch is bootstrap samples, (3/4) is regular samples
parser.add_argument('--denoise_timesteps', type=int,   default=128) # maximum number of diffusion/flow matching steps used during training
parser.add_argument('--grad_clip',         type=float, default=1.0)
parser.add_argument('--ema_decay',         type=float, default=0.999)

# scheduler parameters
parser.add_argument('--scheduler',    type=str,   default='cosine', choices=['cosine', 'linear', 'none'])
parser.add_argument('--warmup_steps', type=int,   default=1000)
parser.add_argument('--lr_min',       type=float, default=1e-7)

# output parameters
parser.add_argument('--save_dir',     type=str, default='batch32-bootstrap4-highclamp')
parser.add_argument('--save_every',   type=int, default=50)
parser.add_argument('--sample_every', type=int, default=25)

# hardware parameters
parser.add_argument('--device',      type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed',        type=int, default=42)

args = parser.parse_args()

# setup
setup_seed(args.seed)
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# device
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
print('running on device', device)

# parse patient lists
train_patients = [int(x) for x in args.train_patients.split(',')]
valid_patients = [int(x) for x in args.valid_patients.split(',')]
test_patients  = [int(x) for x in args.test_patients.split(',')]

print('train patients:', train_patients)
print('valid patients:', valid_patients)
print('test patients:', test_patients)

# parse block out channels
block_out_channels = tuple(int(x) for x in args.block_out_channels.split(','))

# data loaders
print('creating data loaders...')
train_loader, valid_loader, test_loader = create_data_loaders(
    data_path=args.data_path,
    train_patients=train_patients,
    valid_patients=valid_patients,
    test_patients=test_patients,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

# model
print('creating model...')
model = UNet2DModel(
    in_channels=args.input_channels,        # noisy target channel + lr conditioning channel
    out_channels=args.output_channels,      # predicted velocity field
    block_out_channels=block_out_channels,  # feature channels per unet stage
    layers_per_block=args.layers_per_block, # resnet layers per block
    handle_delta_time=True,                 # embed dt for shortcut step size conditioning
    time_embedding_type='positional',       # sinusoidal time embedding
    attention_head_dim=8,                   # attention heads in attn blocks
    norm_num_groups=8,                      # groups for group norm
    dropout=0.1,
    norm_eps=1e-5,
    add_attention=True,                     # add self-attention in mid block
    sample_size=64,                         # spatial resolution
    down_block_types=(
        'DownBlock2D',                      # regular resnet downsampling
        'DownBlock2D',
        'AttnDownBlock2D',                  # resnet downsampling with spatial self-attention
        'AttnDownBlock2D',
    ),
    up_block_types=(
        'AttnUpBlock2D',                    # resnet upsampling with spatial self-attention
        'AttnUpBlock2D',
        'UpBlock2D',                        # regular resnet upsampling
        'UpBlock2D',
    )
)

total_params = sum(p.numel() for p in model.parameters())
print('total parameters: %d' % total_params)

# train
print('starting training...')
train_results = train_shortcut_model(
    model=model,
    train_dataset=train_loader,
    val_dataset=valid_loader,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    bootstrap_every=args.bootstrap_every,
    denoise_timesteps=args.denoise_timesteps,
    weight_decay=args.weight_decay,
    num_epochs=args.num_epochs,
    device=device,
    num_workers=args.num_workers,
    save_dir=str(save_dir),
    save_every=args.save_every,
    seed=args.seed,
    scheduler_type=args.scheduler if args.scheduler != 'none' else None,
    warmup_steps=args.warmup_steps,
    sample_every=args.sample_every,
    lr_min=args.lr_min,
    grad_clip=args.grad_clip,
    ema_decay=args.ema_decay
)

print('best val loss: %.6f' % train_results['best_val_loss'])
