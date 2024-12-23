from ..preprocess.create_box_for_medsam import create_bbox
import numpy as np
import wandb
import logging
from tqdm import tqdm
import os
from medpy import metric
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

list = "/mnt/ssd01_250gb/juny/vfss/SAMed/lists/lists_vfss/test.txt"
path = "/mnt/ssd03_4tb/juny/vfss/vfss_cauh/experiment"

list_file = open(list, 'r')
file_names = list_file.readlines()
file_names = [f.replace('\n', '') for f in file_names]

import wandb
import logging
from tqdm import tqdm

wandb.init(project="vfss_segmentation")
wandb.run.name = "Test of First Run"
wandb.run.save()

# metric_lists = 0.0

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


MedSAM_CKPT_PATH = "/mnt/ssd01_250gb/juny/vfss/SAMed/checkpoints/medsam_vit_b.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()

classes = 1
split = 'test'
metric_list = []
for file_name in file_names:
    data = np.load(os.path.join(path, file_name+'.npz'))
    nonz = np.nonzero(data['label'])
    bbox = create_bbox(data['label'])
    
    img_np = data['image']
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = np.transpose(img_np, (1,2,0))
    H, W, _ = img_3c.shape

    #%% image preprocessing and model inference
    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # box_np = np.array([[95,255, 190, 350]])
    box_np = np.expand_dims(np.array(bbox), 0)
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

    metric_list.append(calculate_metric_percase(medsam_seg, data['label']))

    wandb.log({
            split+' image': wandb.Image(data['image'].transpose(1,2,0)),
            split+' prediction': wandb.Image(medsam_seg),
            split+' groundtruth' : wandb.Image(data['label']),
        })

    # metric_lists += np.array(metric_list)
metric_list = np.array(metric_list)
logging.info('mean_dice %f mean_hd95 %f' % (
    np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1]))
# metric_list = metric_lists / len(file_names)
# for i in range(1, classes + 1):
#     try:
#         # logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))
#         logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
#     except:
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
performance = np.mean(metric_list, axis=0)[0]
mean_hd95 = np.mean(metric_list, axis=0)[1]
logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
logging.info("Testing Finished!")
wandb.log({'mean_dice': performance,
        'mean_hd95': mean_hd95})