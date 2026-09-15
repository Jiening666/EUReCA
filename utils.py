try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import numpy as np
import torch

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if len(image_numpy.shape) == 2:
            image_numpy = image_numpy[np.newaxis,:]
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = (image_numpy - image_numpy.min()) / (image_numpy.max() - image_numpy.min()) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def get_logger_images(visuals, caption_suffix=""):
    """Combine visuals into one wandb.Image.

    caption_suffix: optional annotation appended to the caption, e.g.
    "| N=37 views" so each logged reconstruction records how many
    projection angles produced it.
    """
    full_name = ""
    data = []
    for name, vis in visuals.items():
        full_name += name + " "
        image_numpy = tensor2im(vis)
        data.append(image_numpy)

    combined_array = np.concatenate(data, axis=1)
    result = wandb.Image(
        combined_array,
        caption = (full_name + caption_suffix).strip()
    )
    return result