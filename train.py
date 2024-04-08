import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
from eval import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from util.dataloaders import get_loaders,gen_changelabel
from util.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from main_model import ChangeDetection
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
from tensorboardX import SummaryWriter
from util.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity

def train(opt):
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  

    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs()
    writer = SummaryWriter(save_path)
    save_results = SaveResult(result_save_path)
    save_results.prepare()
    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)
    model = ChangeDetection(opt).cuda()
    seg_criterion = CrossEntropyLoss2d(ignore_index=0) 
    criterion_sc = ChangeSimilarity().cuda()


    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10}, 
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]  
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.001)
    scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)  

    best_metric = 0
    train_avg_loss = 0
    total_bs = 16
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(train_tbar):
            # print(_)
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            if epoch == 0 and i < 20:
                save_results.save_first_batch(batch_img1, batch_img2, batch_label1, batch_label2, i)
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            labels_A = batch_label1.long().cuda()
            labels_B= batch_label2.long().cuda()
            labels_bn = (labels_A > 0).unsqueeze(1).cuda().float()
        
            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))  
            outs = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)
            outputs_A, outputs_B, out = outs


            loss_seg = seg_criterion(outputs_A, labels_A) * 0.5 +  seg_criterion(outputs_B, labels_B) * 0.5
            loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], labels_bn)
            loss_bn =  weighted_BCE_logits(out, labels_bn)
            
            loss = loss_seg + loss_bn + loss_sc
   
            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)

            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label1, batch_label2, labels_bn

        scheduler.step()
        dropblock_step(model)

        Fscd, IoU_mean, Sek, Acc = eval_for_metric(model, val_loader, input_size=opt.input_size, num_classes=opt.num_classes)

        refer_metric =  IoU_mean
        underscore = "_"
        if refer_metric.mean() > best_metric:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            torch.save(model, best_ckp_file)
            best_metric = refer_metric.mean()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(Fscd, IoU_mean, Sek, Acc, refer_metric, best_metric, lr, epoch)
        writer.add_scalar('val_Fscd', Fscd, epoch)
        writer.add_scalar('IoU_mean', IoU_mean, epoch)
        writer.add_scalar('Sek', Sek, epoch)
        writer.add_scalar('Acc', Acc, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')


    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str, default="") 
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--dataset-dir", type=str, default="SECOND-CD/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--finetune", type=bool, default=True)
  

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)
  

    train(opt)
