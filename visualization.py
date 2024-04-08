import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
from torch.nn import functional as F
from skimage import io, exposure
from PIL import Image

from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
from util.utils import accuracy, SCDD_eval_all, AverageMeterSCD
from main_model import ChangeDetection
from util import RS_ST as RS
from torch.utils.data import DataLoader

num_classes = 7
ST_COLORMAP = np.array([(255, 255, 255), (0, 0, 255), (128, 128, 128),
                        (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0)])
color_map = ST_COLORMAP

def label2rgb(gray_label):
    num_classes = color_map.shape[0]
    rgb_label = np.zeros(
        shape=(gray_label.shape[0], gray_label.shape[1], 3), dtype=np.uint8)

    for i in range(num_classes):
        index = np.where(gray_label == i)
        rgb_label[index] = color_map[i]
    return rgb_label

def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ChangeDetection(opt).cuda()
    for ckp_path in opt.ckp_paths:
        if os.path.isdir(ckp_path):
            weight_file = os.listdir(ckp_path)
            ckp_path = os.path.join(ckp_path, weight_file[0])
        print("--Load model: {}".format(ckp_path))
        model = torch.load(ckp_path, map_location=device)
    eval_loader = get_eval_loaders(opt)
    model.eval()

    predict(model, eval_loader, opt.pred_dir, index_map=False, intermediate=False)

def predict(net, eval_loader, pred_dir, index_map=False, intermediate=False):
    pred_A_dir_rgb = os.path.join(pred_dir, 'labelA_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'labelB_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)

    scale = ScaleInOutput(opt.input_size)
    eval_tbar = tqdm(eval_loader)
    for i, (batch_img1, batch_img2, batch_label1, batch_label2, mask_name) in enumerate(eval_tbar):
        mask_name = str(mask_name)[2:][:-3]
       
        batch_img1 = batch_img1.cuda().float()
        batch_img2 = batch_img2.cuda().float()

        with torch.no_grad():
            # batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2)) 
            outs = net(batch_img1, batch_img2)
            outputs_A, outputs_B, out_change = outs
            out_change = F.sigmoid(out_change)

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()

        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()
        
        pred_A = (pred_A * change_mask.long()).numpy()
        pred_B = (pred_B * change_mask.long()).numpy()

        pred_A = Image.fromarray(RS.Index2Color(pred_A).astype('uint8'))
        pred_B = Image.fromarray(RS.Index2Color(pred_B).astype('uint8'))
        
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)

        pred_A.save(pred_A_path, quality=95, dpi=(300, 300))
        pred_B.save(pred_B_path, quality=95, dpi=(300, 300))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection visualization')

    parser.add_argument("--ckp-paths", type=str,
                        default=["./runs/train/5/best_ckp/",])
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--cuda", type=str, default="0")  
    parser.add_argument("--dataset-dir", type=str, default="SECOND-CD/")
    parser.add_argument('--pred_dir', required=False, default='Outimgs/', help='directory to output masks')
    parser.add_argument('--test_dir', required=False, default='SECOND-Bi/val/', help='directory to test images')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--pretrain", type=str, default="") 

    opt = parser.parse_args()
    eval(opt)
