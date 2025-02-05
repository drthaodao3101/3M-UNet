import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
import time
from engine_vofo import *
import os
import sys
from pathlib import Path

from utils import *
from configs.config_setting_vofo import setting_config
import argparse
import warnings
warnings.filterwarnings("ignore")

from pdb import set_trace as st

def main(config):
    parser = argparse.ArgumentParser()           
    parser.add_argument('--mode',type=str,default="train",help="Run mode, can be one of these values:train,val,test (default: 'train')")
    parser.add_argument('--visualize', action='store_true', help="Enable visualization (default: False)")
    args = parser.parse_args()


    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, data='train')
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, data='val')
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                #drop_last=False                #ntcong
                                )
    test_dataset = NPY_datasets(config.data_path, config, data='test')
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                #drop_last=True                 #ntcong
                                )

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
        
    else: raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, config.input_size_h,config.input_size_w, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    elapsed_time={
        "train + val time":0.0,                             
        "train + val pure time":0.0,
        "train time":0.0,                             
        "train pure time":0.0,                        
        "val time":0.0,                             
        "val pure time":0.0,                        
        "test time":0.0,
        "test time-fps":0.0,
        "test pure time":0.0,
        "test pure time-fps":0.0,
    }

    if args.mode=="train":

        if os.path.exists(resume_model):
            print('#----------Resume Model and Other params----------#')
            checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            saved_epoch = checkpoint['epoch']
            start_epoch += saved_epoch
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
            logger.info(log_info)


        step = 0
        print('#----------Training----------#')


        for epoch in range(start_epoch, config.epochs + 1):

            torch.cuda.empty_cache()
            start_time = time.time()                                                        #Train time
            step = train_one_epoch(
                train_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                epoch,
                step,
                logger,
                config,
                elapsed_time,
                writer
            )
            end_time = time.time()                                                          #Train time
            elapsed_time['train time']=elapsed_time['train time']+(end_time-start_time)     #Train time

            start_time = time.time()                                                        #Val time
            loss = val_one_epoch(
                    val_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config,
                    elapsed_time
                )
            end_time = time.time()                                                          #Val time
            elapsed_time['val time']=elapsed_time['val time']+(end_time-start_time)         #Val time

            # if isinstance(loss, tuple):
                # loss = max(loss)  # Lấy giá trị lớn nhất từ tuple
            if loss < min_loss:
                print("Saving model with best loss: ", loss)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
                min_loss = loss
                min_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'latest.pth')) 

    elif args.mode == "test":
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            print('#----------Testing----------#')
            best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
            model.load_state_dict(best_weight)
            
            start_time = time.time()                                                                        #Test time
            _,_,_,_, visualize_name, visualize_ouput = test_one_epoch(
                    test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    elapsed_time,
                    args.visualize
                )
            
            if args.visualize == True:
                #Save the mask output to workspace/SemSeg_VoFo/VM-UNet/results_vofo/vmunet_vofo_Tuesday_31_December_2024_08h_40m_31s/visualize directory
                visualize_dir_path=os.path.join(config.work_dir,"visualize")
                print(f"Visualize: saving ouput in {visualize_dir_path}")
                folder_path = Path(visualize_dir_path)
                folder_path.mkdir(parents=True, exist_ok=True)
                #Get name without extension
                visualize_name_file=[ os.path.splitext(os.path.basename(p))[0] for p in visualize_name]
                assert len(visualize_ouput)==len(visualize_name_file)
                color_palette=config.color_palette
                for index in np.arange(len(visualize_ouput)):
                    mask_output=visualize_ouput[index]
                    save_path=f"{folder_path}/{visualize_name_file[index]}.png"
                    image_path=visualize_name[index]
                    #print(f"{index}-{save_path}")
                    visualize_mask(mask_output, save_path, color_palette,image_path,config.opacity)
                    
                
            end_time = time.time()                                                                          #Test time
            elapsed_time['test time']=elapsed_time['test time']+(end_time-start_time)                       #Test time
            elapsed_time['test time-fps']=test_dataset.__len__()/elapsed_time['test time']                  #Test time fps
            elapsed_time['test pure time-fps']=test_dataset.__len__()/elapsed_time['test pure time']    #Test pure time fps
            
            # os.rename(
            #     os.path.join(checkpoint_dir, 'best.pth'),
            #     os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            # )      
    else:
        raise ValueError("The argument 'mode' is invalid!")
    
    print('#----------Time Summary----------#')
    print(f"Train time only:{elapsed_time['train time']} seconds.")
    print(f"Train pure time only:{elapsed_time['train pure time']} seconds.")
    print(f"Val time only:{elapsed_time['val time']} seconds.")
    print(f"Val pure time only:{elapsed_time['val pure time']} seconds.")
    elapsed_time['train + val time']=elapsed_time['train time']+elapsed_time['val time']
    print(f"Train+Val time:{elapsed_time['train + val time']} seconds.")
    elapsed_time['train + val pure time']=elapsed_time['train pure time']+elapsed_time['val pure time']
    print(f"Train+Val pure time:{elapsed_time['train + val pure time']} seconds.")
    print(f"Test time:{elapsed_time['test time']} seconds ({elapsed_time['test time-fps']} fps).")
    print(f"Test pure time:{elapsed_time['test pure time']} seconds ({elapsed_time['test pure time-fps']} fps).")

if __name__ == '__main__':
    config = setting_config
    main(config)
