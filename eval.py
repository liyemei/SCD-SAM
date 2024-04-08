import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.nn.functional as F

from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
from util.utils import accuracy, SCDD_eval_all, AverageMeterSCD
from main_model import ChangeDetection
import time


def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()  
    save_path, result_save_path = check_eval_dirs()

    save_results = SaveResult(result_save_path)
    save_results.prepare()
    model = ChangeDetection(opt).cuda()
    for ckp_path in opt.ckp_paths:
        if os.path.isdir(ckp_path):
            weight_file = os.listdir(ckp_path)
            ckp_path = os.path.join(ckp_path, weight_file[0])
        print("--Load model: {}".format(ckp_path))
        model = torch.load(ckp_path, map_location=device)

    eval_loader = get_eval_loaders(opt)

    Fscd, IoU_mean, Sek, Acc = eval_for_metric(model, eval_loader)

    save_results.show(Fscd, IoU_mean, Sek, Acc)

def eval_for_metric(model, eval_loader, input_size=512, num_classes=7):  
    running_metricsA = RunningMetrics(num_classes)
    running_metricsB = RunningMetrics(num_classes)
    acc_meter = AverageMeter()

    avg_loss = 0
    scale = ScaleInOutput(input_size)

    model.eval()
    preds_all = []
    labels_all = []
    start = time.time()
    with torch.no_grad():
        torch.cuda.empty_cache()
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            labels_A = batch_label1.long().cuda()
            labels_B = batch_label2.long().cuda()

            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            outs = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)
            outputs_A, outputs_B, out = outs

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out).cpu().detach() > 0.5

            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)

            preds_A = (preds_A * change_mask.squeeze().long()).numpy()
            preds_B = (preds_B * change_mask.squeeze().long()).numpy()

            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_meter.update(acc)

            running_metricsA.update(labels_A, preds_A)
            running_metricsB.update(labels_B, preds_B)

        Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, num_classes)
        curr_time = time.time() - start

        print('%.1fs  Fscd: %.2f mIoU: %.2f Sek: %.2f Accuracy: %.2f' \
              % (curr_time, Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

        scoreA = running_metricsA.get_scores()
        scoreB = running_metricsB.get_scores()
        iouA = scoreA['Mean_IoU']
        iouB = scoreB['Mean_IoU']
        iou = (iouA + iouB) / 2

        F1 = (scoreA['F1_1'] + scoreA['F1_1']) / 2

    # return iou,F1,Fscd, IoU_mean, Sek
    return Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')

    parser.add_argument("--ckp-paths", type=str,
                        default=["./runs/train/3/best_ckp/",])
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")

    parser.add_argument("--cuda", type=str, default="0") 
    parser.add_argument("--dataset-dir", type=str, default="SECOND-CD/")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--pretrain", type=str, default="")  
    parser.add_argument("--loss", type=str, default="bce+dice")

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)