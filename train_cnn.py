import os
import sys
import torch
import argparse
import logging
import numpy as np
from torch.nn.parallel import DistributedDataParallel, DataParallel
from tqdm import tqdm

import utils.comm as comm
from dataset import HSI_LiDAR_Patch_Dataset
from model import HSI_CNNs, MultiModal

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def build_dataloader(args, split='train'):
    assert split in ['train', 'test', 'val']
    if split == 'train':
        hsi_path = os.path.join(args.data_path, 'HSI_TrTe/Patch_HSI_TrSet.mat')
        lidar_path = os.path.join(args.data_path, 'LiDAR_TrTe/Patch_LiDAR_TrSet.mat')
        label_path = os.path.join(args.data_path, 'Patch_TrLabel.mat')
    elif split == 'test':
        hsi_path = os.path.join(args.data_path, 'HSI_TrTe/Patch_HSI_TeSet.mat')
        lidar_path = os.path.join(args.data_path, 'LiDAR_TrTe/Patch_LiDAR_TeSet.mat')
        label_path = os.path.join(args.data_path, 'Patch_TeLabel.mat')
    else:
        assert NotImplementedError
    
    logger.info("Building {} dataset from {}...".format(split, args.data_path))
    hsi_lidar_dataset = HSI_LiDAR_Patch_Dataset(hsi_path, lidar_path, label_path, split)
    dataloader = torch.utils.data.DataLoader(
        hsi_lidar_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )

    return dataloader
    
def build_model(args):
    # build model
    model = HSI_CNNs(args.num_classes)
    # move to GPU
    model = DataParallel(model)
    model.to(args.device)
    return model

def build_optimizer(args, model):
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    return optimizer

def build_lr_scheduler(args, optimizer):
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[10, 20, 30], gamma=0.1
    # )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    return scheduler

def do_train(args, model, resume=False):
    if resume:
        raise NotImplementedError
    model.train()
    data_loader = build_dataloader(args, split='train')
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    start_epoch = 0
    max_epoch = args.num_epoch
    num_samples = len(data_loader.dataset)
    # noise debug
    # x = torch.randn(2, 144, 7, 7).to(args.device)
    # y = model(x)
    train_acc = []
    test_acc = []
    logger.info("Starting training from epoch {}".format(start_epoch))
    for epoch in tqdm(range(start_epoch, max_epoch)):
        acc_sum = 0
        loss_sum = 0
        for data, label in tqdm(data_loader, leave=False):
            # move data dict's each key to GPU
            label = label.type(torch.LongTensor)
            label = label.to(args.device)
            # for key in data.keys():
            #     data[key] = data[key].to(args.device)
                
            optimizer.zero_grad()
            x = data['hsi_data'].to(args.device)
            pred = model(x)
            loss = loss_fn(pred, label)
            prob = torch.softmax(pred, dim=-1)
            acc_iter = (prob.argmax(dim=-1) == label.int()).sum()
            
            loss.backward()
            optimizer.step()
            
            acc_sum += acc_iter.item()
            loss_sum += loss.item()
        
        scheduler.step()
        loss_normalizer = len(data_loader.dataset)//args.batch_size
        loss_avg = loss_sum / loss_normalizer
        train_acc_epoch = acc_sum / num_samples * 100
        logger.info("\n [Train] Epoch: {}, Loss: {:.2f}, Acc: {:.2f} \n".format(epoch, loss_avg, train_acc_epoch))
        # if epoch % 10 == 0 and epoch != 0:
        test_acc_epoch, pred = do_test(args, model)  
        train_acc.append(train_acc_epoch)
        test_acc.append(test_acc_epoch)
        
        comm.synchronize()
        
    return train_acc, test_acc, pred

@torch.no_grad()     
def do_test(args, model):
    model.eval()
    data_loader = build_dataloader(args, split='test')
    acc_sum = 0
    loss_sum = 0
    pred_all = []
    for data, label in tqdm(data_loader):
        label = label.type(torch.LongTensor)
        label = label.to(args.device)
        for key in data.keys():
            data[key] = data[key].to(args.device)
        pred = model(data['hsi_data'])
        pred_all.append(torch.softmax(pred, dim=-1).argmax(dim=1).cpu().numpy())
        acc_iter = (torch.softmax(pred, dim=-1).argmax(dim=1) == label).sum().item()
        acc_sum += acc_iter
    
    acc = acc_sum / len(data_loader.dataset) * 100
    logger.info("\n [Test] Acc: {:.2f} \n".format(acc))
    model.train()
    return acc, pred_all
    
def main(args):
    logger.info("Dataset file from {}...".format(args.data_path)) 
    # build model 
    model = build_model(args)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        model.load_state_dict(torch.load(args.weight_path))
        return do_test(args, model)
    
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        
    # train model 
    train_acc, test_acc, pred = do_train(args, model, resume=args.resume)
    np.save('train_acc.npy', train_acc)
    np.save('test_acc.npy', test_acc)
    return do_test(args, model)  

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/MultimodalRS/HS-LiDAR', help='Path to data')
    # Training parameters
    parser.add_argument('--num_epoch', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--eval_only', action='store_true', help='Evaluate the model')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--weight_path', type=str, default='weights', help='Path to save weights')
    # Model parameters
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'], help='Model name')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add', 'mul'], help='Fusion method')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes')
    parser.add_argument('--seed', type=int, default=1, help='Number of bands')
    parser.add_argument('--infer_type', type=str, default='MML', choices=['MML', 'CML'], help='Multi-modal inference type')
    return parser.parse_args()
    
if __name__ == '__main__':
    # sys.argv = ['train.py',
    #     '--data_path', './data/MultimodalRS/HS-LiDAR',
    # ]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    args.device = device
    set_seed(args)
    main(args)
