import torch
import numpy as np
import math

def get_targets(batch_size, bootstrap_every, denoise_timesteps, train_state, images, key=0, force_t=-1, force_dt=-1, use_ema=False):
    # generate training targets for shortcut model training:
    #   1) flow matching - train model with velocity (x_1 - x_0)
    #   2) shortcut bootstrap - train with two-step velocity estimate so the model learns to generate with few steps
    # bootstrap_every controls split: 1/bootstrap_every of batch uses shortcut targets, rest uses flow matching

    time_seed  = key + 1
    noise_seed = key + 2

    info = {}

    # shortcut / bootstrap section
    # this section generates targets for the shortcut portion of the batch.
    # the goal is to teach the model to jump from t to t+dt in one step,
    # where dt can be large (1/2 the full trajectory) rather than a tiny diffusion step.

    bootstrap_batchsize = batch_size // bootstrap_every
    log2_sections = int(math.log2(denoise_timesteps))

    # select samples across all step sizes so the model learns all of them
    dt_base = torch.repeat_interleave(
        log2_sections - 1 - torch.arange(log2_sections),
        bootstrap_batchsize // log2_sections
    )
    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0])])

    force_dt_vec = torch.ones(bootstrap_batchsize) * force_dt
    dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base)
    dt_base = dt_base.to(images.device)

    dt = 1 / (2 ** dt_base)       # actual step size, e.g. dt_base=2 -> dt=0.25
    dt_base_bootstrap = dt_base + 1  # half-step dt_base used for the two sub-steps
    dt_bootstrap      = dt / 2       # half of the full step size

    # sample a random starting time t within each step size section.
    # t represents where along the noisy trajectory (0= gaussian noise, 1=clean image)
    dt_sections = 2 ** dt_base

    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float()
        for val in dt_sections.cpu()
    ])
    t = t / dt_sections.cpu()  # normalize to [0, 1]

    force_t_vec = torch.ones(bootstrap_batchsize, dtype=torch.float32) * force_t
    t = torch.where(force_t_vec != -1, force_t_vec, t)
    t = t.to(images.device)
    t_full = t[:, None, None, None]  # expand for broadcasting over H x W

    # split image channels
    bootstrap_images  = images[:bootstrap_batchsize]  # B x 2 x H x W
    x_1               = bootstrap_images[:, 0:1]      # target metabolic map
    condition_channel = bootstrap_images[:, 1:2]      # lr conditioning metabolic map

    # create noisy image x_t by linearly interpolating between gaussian noise (x_0) and HR metabolic map (x_1).
    torch.manual_seed(noise_seed)
    x_0        = torch.randn_like(x_1)
    x_t_target = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1 # 1e-5 helps keep training stable
    x_t        = torch.cat([x_t_target, condition_channel], dim=1)

    # generate bootstrap targets for a large step dt using two smaller half-steps
    # step 1: predict velocity at (x_t, t) using half-step dt
    with torch.no_grad():
        if use_ema:
            v_b1 = train_state.call_model_ema(x_t, t, dt_base_bootstrap)
        else:
            v_b1 = train_state.call_model(x_t, t, dt_base_bootstrap, train=False)

    # step 2: take a half step forward using v_b1, then predict velocity at the new position
    t2   = t + dt_bootstrap
    x_t2 = torch.cat([
        torch.clamp(x_t_target + dt_bootstrap[:, None, None, None] * v_b1, -4, 4), #clamp at -4, 4 to ensure velocities are plausible
        condition_channel
    ], dim=1)

    with torch.no_grad():
        if use_ema:
            v_b2 = train_state.call_model_ema(x_t2, t2, dt_base_bootstrap)
        else:
            v_b2 = train_state.call_model(x_t2, t2, dt_base_bootstrap, train=False)

    v_target = torch.clamp((v_b1 + v_b2) / 2, -4, 4)

    bst_v  = v_target
    bst_dt = dt_base
    bst_t  = t
    bst_xt = x_t

    # flow matching section
    # standard flow matching (model learns velocity field from gaussian noise to HR metabolic map)

    target_channel_flow    = images[:, 0:1]  # B x 1 x H x W
    condition_channel_flow = images[:, 1:2]  # B x 1 x H x W

    t_flow = torch.randint(
        low=0,
        high=denoise_timesteps,
        size=(target_channel_flow.shape[0],),
        dtype=torch.float32,
        device=images.device
    )
    t_flow = t_flow / denoise_timesteps  # normalize to [0, 1]

    force_t_vec_flow = torch.ones(target_channel_flow.shape[0], device=images.device) * force_t
    t_flow      = torch.where(force_t_vec_flow != -1, force_t_vec_flow, t_flow)
    t_flow_full = t_flow[:, None, None, None]

    # interpolate between noise and HR metabolic map
    torch.manual_seed(noise_seed)
    x_0_flow = torch.randn_like(target_channel_flow)
    x_1_flow = target_channel_flow
    x_t_flow = (1 - (1 - 1e-5) * t_flow_full) * x_0_flow + t_flow_full * x_1_flow
    x_t_flow = torch.cat([x_t_flow, condition_channel_flow], dim=1)

    v_t_flow = x_1_flow - (1 - 1e-5) * x_0_flow

    dt_flow      = int(math.log2(denoise_timesteps))
    dt_base_flow = torch.ones(target_channel_flow.shape[0], dtype=torch.int32, device=images.device) * dt_flow

    # merge bootstrap and flow matching targets
    bst_size      = batch_size // bootstrap_every
    bst_size_data = batch_size - bst_size

    x_t_combined     = torch.cat([bst_xt, x_t_flow[:bst_size_data]],    dim=0)
    t_combined        = torch.cat([bst_t,  t_flow[:bst_size_data]],       dim=0)
    dt_base_combined  = torch.cat([bst_dt, dt_base_flow[:bst_size_data]], dim=0)
    v_t_combined      = torch.cat([bst_v,  v_t_flow[:bst_size_data]],     dim=0)

    info['bootstrap_ratio']       = torch.mean((dt_base_combined != dt_flow).float())
    info['v_magnitude_bootstrap'] = torch.sqrt(torch.mean(torch.square(bst_v)))
    info['v_magnitude_b1']        = torch.sqrt(torch.mean(torch.square(v_b1)))
    info['v_magnitude_b2']        = torch.sqrt(torch.mean(torch.square(v_b2)))

    return x_t_combined, v_t_combined, t_combined, dt_base_combined, info