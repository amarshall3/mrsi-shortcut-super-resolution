import torch
import numpy as np

def downsample_image(image, factor):
    image_np = image.cpu().numpy()
    rows, cols = image_np.shape[1:]
    crow, ccol = rows // 2, cols // 2

    # transform image to frequency domain
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)

    # create and apply low-pass filter
    lp_mask = np.zeros((rows, cols), dtype=np.uint8)
    lp_mask[crow - int(crow / factor): crow + int(crow / factor), ccol - int(ccol / factor): ccol + int(ccol / factor)] = 1
    fshift_filtered = fshift * lp_mask
    
    # transform image to spatial domain
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
    downsampled_image_np = np.float32(img_back)
    downsampled_image = torch.from_numpy(downsampled_image_np).to(image.device)
    return downsampled_image
