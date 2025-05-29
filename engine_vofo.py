from typing import *
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
import time
from scipy.spatial.distance import cdist
from medpy import metric
import miseval

from utils import save_imgs
from SurfaceDistance.surface_distance import metrics
from pdb import set_trace as st
import torch.nn.functional as F

def deep_supervision_loss(out,out_ds,targets,criterion,weight):
    loss=criterion(out, targets)*weight[-1]             #final_output 
    targets=targets.unsqueeze(1)
    targets = F.interpolate(targets, scale_factor=0.5, mode='nearest')
    for i in range(len(out_ds)-1, 2, -1):
        loss=loss+(criterion(out_ds[i], targets.squeeze(1))*weight[i])
        targets = F.interpolate(targets, scale_factor=0.5, mode='nearest')
    return loss



def calculate_metric_percase(pred, gt):
    spacing=(1.0, 1.0)
    tolerance=1.0
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    iou = metric.binary.jc(pred, gt)
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = metric.binary.hd95(pred, gt)

        pred_points = np.argwhere(pred)
        gt_points = np.argwhere(gt)
        if len(pred_points) > 0 and len(gt_points) > 0:
            distances = cdist(pred_points, gt_points)
            min_distances = np.min(distances, axis=1)
            nsd = np.sum(min_distances < tolerance) / len(min_distances)
        else:
            nsd = 0  # Avoid division by zero
        
        return dice, hd95, iou, nsd
    
    elif pred.sum() > 0 and gt.sum() == 0:
        hd95 = np.nan  # No ground truth to compare to
        nsd = 0  # No meaningful NSD calculation
        return dice, hd95, iou, nsd
    
    elif pred.sum() == 0 and gt.sum() > 0:
        hd95 = np.nan  # No prediction to compare to
        nsd = 0  # No meaningful NSD calculation
        return dice, hd95, iou, nsd
    
    else:
        hd95 = 0  # No distance since both are empty
        nsd = 1  # Perfect match when both are empty
        return dice, hd95, iou, nsd
    


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    elapsed_time,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 

    loss_list = []

    for iter, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        step += iter
        model.zero_grad()
        images, targets, _ = data

        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        
        start_time = time.time()                                                        #Train pure time
        out,out_ds = model(images) # Shape: bs x num_classes x H x W
        end_time = time.time()                                                          #Train pure time
        elapsed_time['train pure time']=elapsed_time['train pure time']+(end_time-start_time)     #Train time

        # Thay đổi kích thước của targets
        targets = targets.squeeze(1)  # Loại bỏ chiều có kích thước 1
        # Shape: bs x H x W, classes: [0., 1., 2., 3., 4., 5., 6.]]
        if config.deep_supervision:
            loss=deep_supervision_loss(out, out_ds, targets, criterion,config.deep_supervision_weight)
        else:
            loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step

def val_one_epoch(val_loader, model, criterion, epoch, logger, config,elapsed_time, visualize:bool=False) -> Union[float, Tuple[float, List[str], List[np.ndarray], List[np.ndarray]]]:
    model.eval()
    preds = []
    gts = []
    visualize_name = []
    loss = torch.zeros(1, device="cuda")
    metric_list_per_class = {}
    num_classes = config.num_classes

    stime = time.time()

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk, path = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).float(),
            )

            start_time = time.time()                                                        #Val pure time
            out,out_ds = model(img)
            end_time = time.time()                                                          #Val pure time
            elapsed_time['val pure time']=elapsed_time['val pure time']+(end_time-start_time)         #Val pure time

            # Resize targets to remove channel dimension
            msk = msk.squeeze(1)  # Shape: [B, H, W]

            # loss += criterion(out, msk)
            if config.deep_supervision:
                loss += deep_supervision_loss(out, out_ds, msk, criterion,config.deep_supervision_weight)
            else:
                loss += criterion(out, msk)

            # Store predictions and ground truths
            pred = torch.argmax(out, dim=1).cpu().detach().numpy()  # Shape: [BS,H, W]
            msk = msk.cpu().detach().numpy()  # Shape: [BS,H, W]

            #store for visualize
            if visualize==True:
                preds.append(pred.squeeze())
                gts.append(msk.squeeze())
                visualize_name.extend(path)


            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, hd95, iou, nsd = calculate_metric_percase(pred_class, gt_class)
                if i not in metric_list_per_class:
                    metric_list_per_class[i] = []
                metric_list_per_class[i].append([dice, hd95, iou, nsd])


    metric_list = []
    for i in range(1, num_classes):
        metric_list_per_class[i] = np.array(metric_list_per_class[i])
        # metric_list_per_class[i] = metric_list_per_class[i].sum(axis=0).astype(float) / len(test_loader)
        metric_list_per_class[i] = np.nanmean(metric_list_per_class[i], axis=0)
        metric_list.append([metric_list_per_class[i][0], metric_list_per_class[i][1], metric_list_per_class[i][2], metric_list_per_class[i][3]])
    
    mean_dice = [metric_list[i][0] for i in range(len(metric_list))]  # Mean dice
    mean_hd95 = [metric_list[i][1] for i in range(len(metric_list))]  # Mean hd95

    all_class_mean_dice = np.mean(mean_dice)
    all_class_mean_hd95 = np.mean(mean_hd95)

    # np.nanmean(np.array(metric_list), axis=0)

    for i in range(1, num_classes): 
        print(f"Val epoch: {epoch}; Class {i} mean_dice: {mean_dice[i-1]:.4f}, mean_hd95: {mean_hd95[i-1]:.4f}")
    
    print(f"Val epoch: {epoch}; ALL class: mean_dice: {all_class_mean_dice:.4f}, mean_hd95: {all_class_mean_hd95:.4f}")

    # return mean_dice, mean_hd95
    loss /= len(val_loader)
    if visualize==True:
        return (loss.item(), visualize_name, preds, gts), all_class_mean_dice, all_class_mean_hd95 
    else:
        return loss.item(), all_class_mean_dice, all_class_mean_hd95


def test_one_epoch(test_loader, model, criterion, logger, config,elapsed_time,visualize, test_data_name=None):
    """
    Testing for one epoch.
    """
    visualize_name=[]
    visualize_ouput=[]

    model.eval()
    preds = []
    gts = []
    loss_list = []
    metric_list_per_class = {}
    num_classes = config.num_classes

    stime = time.time()

    with torch.no_grad():
        case_spacing = [1,1,1]
        for data in tqdm(test_loader):
            img, msk, path = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).long(),
            )

            start_time = time.time()                                                                        #Test pure time
            out,out_ds = model(img)
            end_time = time.time()                                                                          #Test pure time
            elapsed_time['test pure time']=elapsed_time['test pure time']+(end_time-start_time)             #Test pure time

            # Resize targets to remove channel dimension
            msk = msk.squeeze(1)  # Shape: [B, H, W]

            # Store predictions and ground truths
            pred = torch.argmax(out, dim=1).cpu().detach().numpy()  # Shape: [BS,H,W]
            msk = msk.cpu().detach().numpy()  # Shape: [BS,H, W]
            
            #store for visualize
            if visualize==True:
                vis=[pred[i] for i in np.arange(pred.shape[0])]
                visualize_ouput.extend(vis)
                visualize_name.extend(path)


            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, hd95,iou,nsd = calculate_metric_percase(pred_class, gt_class)
                if i not in metric_list_per_class:
                    metric_list_per_class[i] = []
                metric_list_per_class[i].append([dice, hd95,iou,nsd])
                
    metric_list = []
    for i in range(1, num_classes):
        metric_list_per_class[i] = np.array(metric_list_per_class[i])
        # metric_list_per_class[i] = metric_list_per_class[i].sum(axis=0).astype(float) / len(test_loader)
        metric_list_per_class[i] = np.nanmean(metric_list_per_class[i], axis=0)
        metric_list.append([metric_list_per_class[i][0], metric_list_per_class[i][1], metric_list_per_class[i][2], metric_list_per_class[i][3]])

    mean_dice = [metric_list[i][0] for i in range(len(metric_list))]  # Mean dice
    mean_hd95 = [metric_list[i][1] for i in range(len(metric_list))]  # Mean hd95
    mean_iou = [metric_list[i][2] for i in range(len(metric_list))]  # Mean iou
    mean_nsd = [metric_list[i][3] for i in range(len(metric_list))]  # Mean nsd

    for i in range(1, num_classes): 
        print(f"Test epoch; Class {i} mean_dice: {mean_dice[i-1]:.4f}, mean_hd95: {mean_hd95[i-1]:.4f}, mean_iou: {mean_iou[i-1]:.4f}, mean_nsd: {mean_nsd[i-1]:.4f}")
    
    all_class_mean_dice=np.mean(mean_dice)
    all_class_mean_hd95=np.mean(mean_hd95)
    all_class_mean_iou=np.mean(mean_iou)
    all_class_mean_nsd=np.mean(mean_nsd)
    print("ALL class: mean_dice: %.4f, mean_hd95: %.4f, mean_iou: %.4f, mean_nsd: %.4f"% (all_class_mean_dice,all_class_mean_hd95,all_class_mean_iou,all_class_mean_nsd))
    return all_class_mean_dice, all_class_mean_hd95, all_class_mean_iou, all_class_mean_nsd, visualize_name, visualize_ouput 

