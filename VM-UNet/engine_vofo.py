import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import time
from pdb import set_trace as st
from medpy import metric
from SurfaceDistance.surface_distance import metrics
from scipy.spatial.distance import cdist
def calculate_metric_percase(pred, gt):
    spacing=(1.0, 1.0)
    tolerance=1.0
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        iou = metric.binary.jc(pred, gt)

        #Tính NSD (Normalized surface distance)                     https://github.com/google-deepmind/surface-distance/blob/master/surface_distance/metrics.py
        #Method 1
        #surface_distances=metrics.compute_surface_distances(gt[0,...]>0, pred[0,...]>0, spacing=(1, 1))
        #distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        #distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        #nsd_gt = np.mean(distances_gt_to_pred <= tolerance) if len(distances_gt_to_pred) > 0 else 0
        #nsd_pred = np.mean(distances_pred_to_gt <= tolerance) if len(distances_pred_to_gt) > 0 else 0
        #nsd = (nsd_gt + nsd_pred) / 2
        
        #Method 2
        #asd = metric.binary.asd(pred, gt)
        #nsd = min(1, asd / tolerance)
        
        #Method 3
        pred_points = np.argwhere(pred)
        gt_points = np.argwhere(gt)
        distances = cdist(pred_points, gt_points)  
        min_distances = np.min(distances, axis=1)  
        nsd = np.sum(min_distances < tolerance) / len(min_distances)  
        
        return dice, hd95, iou, nsd
    elif pred.sum() > 0 and gt.sum()==0:                            #ntcong - there are issue https://github.com/Beckschen/TransUNet/issues/39
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0


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
        out = model(images) # Shape: bs x num_classes x H x W
        end_time = time.time()                                                          #Train pure time
        elapsed_time['train pure time']=elapsed_time['train pure time']+(end_time-start_time)     #Train time

        # Thay đổi kích thước của targets
        targets = targets.squeeze(1)  # Loại bỏ chiều có kích thước 1
        # Shape: bs x H x W, classes: [0., 1., 2., 3., 4., 5., 6.]]
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


# def val_one_epoch(val_loader, model, criterion, epoch, logger, config):
#     """
#     Validation for one epoch.
#     """
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     metric_list = []

#     stime = time.time()

#     with torch.no_grad():
#         for data in tqdm(val_loader):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).long()

#             out = model(img)
#             #print(msk.shape)
#             # Resize targets to remove channel dimension
#             msk = msk.squeeze(1)  # Shape: [B, H, W]
#             #print(msk.shape)
            
#             #print(np.unique(msk.cpu().numpy()))

#             loss = criterion(out, msk)
#             if isinstance(loss, tuple):
#                 loss = loss[0]  # Ensure loss is a scalar
#             loss_list.append(loss.item())
            
#             #print(out.shape)
#             #print(out[0,:,0,0])
#             # Store predictions and ground truths
#             preds.append(torch.argmax(out, dim=1).cpu().detach().numpy())  # Shape: [B, H, W]
#             gts.append(msk.cpu().detach().numpy())

#     # Flatten predictions and ground truths for metric calculation
#     preds = np.concatenate([p.flatten() for p in preds])
#     gts = np.concatenate([g.flatten() for g in gts])

#     # Calculate mean_dice and hd95 metrics without confusion matrix
#     num_classes = config.num_classes
#     for i in range(num_classes):
#         #print(np.unique(gts))
#         #print(np.unique(preds))
#         gt_class = (gts == i).astype(int)
#         pred_class = (preds == i).astype(int)

#         intersection = np.sum(gt_class * pred_class)
#         union = np.sum(gt_class) + np.sum(pred_class)
#         dice = (2 * intersection) / union if union > 0 else 0

#         # Placeholder for hd95 calculation
#         hd95 = np.random.uniform(0, 1)

#         metric_list.append([dice, hd95])

#     metric_list = np.array(metric_list)
#     performance = np.mean(metric_list[:, 0])  # Mean dice
#     mean_hd95 = np.mean(metric_list[:, 1])  # Mean hd95

#     for i in range(num_classes):
#         logger.info('Class %d mean_dice: %f, mean_hd95: %f' % (i, metric_list[i][0], metric_list[i][1]))

#     etime = time.time()
#     log_info = f'val epoch: {epoch}, mean_dice: {performance:.4f}, mean_hd95: {mean_hd95:.4f}, time(s): {etime-stime:.2f}'
#     print(log_info)
#     logger.info(log_info)

#     return performance, mean_hd95

# def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
#     """
#     Testing for one epoch.
#     """
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     metric_list = []

#     stime = time.time()

#     with torch.no_grad():
#         for data in tqdm(test_loader):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).long()

#             out = model(img)

#             # Resize targets to remove channel dimension
#             msk = msk.squeeze(1)  # Shape: [B, H, W]

#             loss = criterion(out, msk)
#             if isinstance(loss, tuple):
#                 loss = loss[0]  # Ensure loss is a scalar
#             loss_list.append(loss.item())

#             # Store predictions and ground truths
#             preds.append(torch.argmax(out, dim=1).cpu().detach().numpy())  # Shape: [B, H, W]
#             gts.append(msk.cpu().detach().numpy())

#     # Flatten predictions and ground truths for metric calculation
#     preds = np.concatenate([p.flatten() for p in preds])
#     gts = np.concatenate([g.flatten() for g in gts])

#     # Calculate mean_dice and hd95 metrics without confusion matrix
#     num_classes = config.num_classes
#     for i in range(num_classes):
#         gt_class = (gts == i).astype(int)
#         pred_class = (preds == i).astype(int)

#         intersection = np.sum(gt_class * pred_class)
#         union = np.sum(gt_class) + np.sum(pred_class)
#         dice = (2 * intersection) / union if union > 0 else 0

#         # Placeholder for hd95 calculation
#         hd95 = np.random.uniform(0, 1)

#         metric_list.append([dice, hd95])

#     metric_list = np.array(metric_list)
#     performance = np.mean(metric_list[:, 0])  # Mean dice
#     mean_hd95 = np.mean(metric_list[:, 1])  # Mean hd95

#     for i in range(num_classes):
#         logger.info('Class %d mean_dice: %f, mean_hd95: %f' % (i, metric_list[i][0], metric_list[i][1]))

#     etime = time.time()
#     log_info = f'Testing {("(" + test_data_name + ")") if test_data_name else ""}, mean_dice: {performance:.4f}, mean_hd95: {mean_hd95:.4f}, time(s): {etime-stime:.2f}'
#     print(log_info)
#     logger.info(log_info)

#     return performance, mean_hd95


def val_one_epoch(val_loader, model, criterion, epoch, logger, config,elapsed_time):
    model.eval()
    preds = []
    gts = []
    loss = torch.zeros(1, device="cuda")
    metric_list = 0.0
    num_classes = config.num_classes

    stime = time.time()

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk, _ = data
            img, msk = (
                img.cuda(non_blocking=True).float(),
                msk.cuda(non_blocking=True).long(),
            )

            start_time = time.time()                                                        #Val pure time
            out = model(img)
            end_time = time.time()                                                          #Val pure time
            elapsed_time['val pure time']=elapsed_time['val pure time']+(end_time-start_time)         #Val pure time

            # Resize targets to remove channel dimension
            msk = msk.squeeze(1)  # Shape: [B, H, W]

            loss += criterion(out, msk)

            # Store predictions and ground truths
            pred = torch.argmax(out, dim=1).cpu().detach().numpy()  # Shape: [BS,H, W]
            msk = msk.cpu().detach().numpy()  # Shape: [BS,H, W]
            result = []
            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, hd95 = calculate_metric_percase(pred_class, gt_class)
                result.append([dice, hd95])

            metric_list += np.array(result)

    metric_list = metric_list / len(val_loader)
    mean_dice = metric_list[:, 0]  # Mean dice
    mean_hd95 = metric_list[:, 1]  # Mean hd95

    for i in range(1, num_classes): 
        print(f"Val epoch: {epoch}; Class {i} mean_dice: {mean_dice[i-1]:.4f}, mean_hd95: {mean_hd95[i-1]:.4f}")

    # return mean_dice, mean_hd95
    loss /= len(val_loader)
    return loss.item()


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
    metric_list = 0.0
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
            out = model(img)
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


            result = []
            for i in range(1, num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice, hd95,iou,nsd = calculate_metric_percase(pred_class, gt_class)

                
                result.append([dice, hd95,iou,nsd])
                
            metric_list += np.array(result)

    metric_list = metric_list / len(test_loader)
    mean_dice = metric_list[:, 0]  # Mean dice
    mean_hd95 = metric_list[:, 1]  # Mean hd95
    mean_iou = metric_list[:, 2]  # Mean iou
    mean_nsd = metric_list[:, 3]  # Mean iou

    for i in range(1, num_classes): print("Class %d mean_dice: %.4f, mean_hd95: %.4f, mean_iou: %.4f, mean_nsd: %.4f"% (i, mean_dice[i-1], mean_hd95[i-1],mean_iou[i-1],mean_nsd[i-1]))

    all_class_mean_dice=np.mean(mean_dice)
    all_class_mean_hd95=np.mean(mean_hd95)
    all_class_mean_iou=np.mean(mean_iou)
    all_class_mean_nsd=np.mean(mean_nsd)
    print("ALL class: mean_dice: %.4f, mean_hd95: %.4f, mean_iou: %.4f, mean_nsd: %.4f"% (all_class_mean_dice,all_class_mean_hd95,all_class_mean_iou,all_class_mean_nsd))
    return all_class_mean_dice, all_class_mean_hd95, all_class_mean_iou, all_class_mean_nsd, visualize_name, visualize_ouput 

