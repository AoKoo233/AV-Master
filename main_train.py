import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import json
import numpy as np

from dataloader import *
from nets.AV_Master_ours import AV_Master
from configs.arguments_AV_Master import parser


import warnings
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')


print("\n--------------- AV_Master Training --------------- \n")

def avqa_dataset(label,args):
    
    val_dataset = AVQA_dataset(label = label, 
                               args = args, 
                               audios_feat_dir = args.audios_feat_dir, 
                               visual_feat_dir = args.visual_feat_dir,
                               clip_vit_b32_dir = args.clip_vit_b32_dir,
                               clip_qst_dir = args.clip_qst_dir,  
                               clip_word_dir = args.clip_word_dir,  
                               transform = transforms.Compose([ToTensor()]), 
                               mode_flag = 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return val_loader


def train(args, model, train_loader, optimizer, criterion, writer, epoch):
    
    model.train()

    for batch_idx, sample in enumerate(train_loader):
        audios_feat, visual_feat, patch_feat, target, question, qst_word = sample['audios_feat'].to('cuda'), sample['visual_feat'].to('cuda'), sample['patch_feat'].to('cuda'), sample['answer_label'].to('cuda'), sample['question'].to('cuda'), sample['qst_word'].to('cuda')
        neg_visual_feat = sample['neg_visual_feat'].to('cuda')
        neg_audios_feat = sample['neg_audios_feat'].to('cuda')

        optimizer.zero_grad()
        output_qa,output_qa2,output_qa3,ctrloss = model(audios_feat, visual_feat, patch_feat, question, qst_word, neg_visual_feat, neg_audios_feat)  

        loss1 = criterion(output_qa, target) + ctrloss
        loss2 = criterion(output_qa2, target)
        loss3 = criterion(output_qa3, target)

        loss = loss1 + loss2 + loss3 + ctrloss


        writer.add_scalar('run/both', loss.item(), epoch * len(train_loader) + batch_idx)

        #loss.backward()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #      epoch, batch_idx * len(audios_feat), len(train_loader.dataset),
            #      100. * batch_idx / len(train_loader), loss.item()),flush=True)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_all: {:.6f}\tLoss_a: {:.6f}\tLoss_v: {:.6f}\tLoss_ctr: {:.6f}'.format(
                  epoch, batch_idx * len(audios_feat), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item(), loss2.item(), loss3.item(), ctrloss.item()),flush=True)


def eval(model, val_loader, writer, epoch):
    
    model.eval()
    total_qa = 0
    correct_qa = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audios_feat, visual_feat, patch_feat, target, question, qst_word = sample['audios_feat'].to('cuda'), sample['visual_feat'].to('cuda'), sample['patch_feat'].to('cuda'), sample['answer_label'].to('cuda'), sample['question'].to('cuda'), sample['qst_word'].to('cuda')

            preds_qa,preds_qa2,preds_qa3,_ = model(audios_feat, visual_feat, patch_feat, question, qst_word)

            _, predicted = torch.max(preds_qa.data + preds_qa2.data + preds_qa3.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Current Acc: %.2f %%' % (100 * correct_qa / total_qa),flush=True)
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa



def main():

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    tensorboard_name = args.checkpoint
    writer = SummaryWriter('runs/strn/' + TIMESTAMP + '_' + tensorboard_name)

    model = AV_Master(args).to('cuda')
    model = nn.DataParallel(model).to('cuda')

    train_dataset = AVQA_dataset(label = args.label_train, 
                                 args = args, 
                                 audios_feat_dir = args.audios_feat_dir, 
                                 visual_feat_dir = args.visual_feat_dir,
                                 clip_vit_b32_dir = args.clip_vit_b32_dir,
                                 clip_qst_dir = args.clip_qst_dir, 
                                 clip_word_dir = args.clip_word_dir, 
                                 transform = transforms.Compose([ToTensor()]), 
                                 mode_flag = 'train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    

    val_dataset = AVQA_dataset(label = args.label_val, 
                               args = args, 
                               audios_feat_dir = args.audios_feat_dir, 
                               visual_feat_dir = args.visual_feat_dir,
                               clip_vit_b32_dir = args.clip_vit_b32_dir,
                               clip_qst_dir = args.clip_qst_dir,  
                               clip_word_dir = args.clip_word_dir,  
                               transform = transforms.Compose([ToTensor()]), 
                               mode_flag = 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    av_avg_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_avg.json', args)
    av_temp_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_temp.json', args)
    av_comp_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_comp.json', args)
    av_local_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_local.json', args)
    av_count_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_count.json', args)
    av_exist_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_av_exist.json', args)

    a_avg_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_a_avg.json', args)
    a_comp_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_a_comp.json', args)
    a_count_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_a_count.json', args)

    v_avg_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_v_avg.json', args)
    v_local_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_v_local.json', args)
    v_count_loader = avqa_dataset('./dataset/split_que_id/sub_val/music_avqa_val_v_count.json', args)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    #model.load_state_dict(torch.load('models_pstpPSTP_Net.pt'),strict=True)


    best_acc = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        train(args, model, train_loader, optimizer, criterion, writer, epoch=epoch)

        # evaluate on validation set
        scheduler.step(epoch)
        print("\n--------------- AV_Master --------------- \n")
        print("Len of Test_loader: ", len(val_loader.dataset))
        current_acc = eval(model, val_loader, writer, epoch)
        if current_acc >= best_acc:
            best_acc = current_acc
            best_epoch = epoch
            torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
        
        print("Best Acc: %.2f %%"%best_acc)
        print("Best Epoch: ", best_epoch)
        print("*"*20)

        print("\n--------------- AV --------------- \n")
        print("Len of avg_loader: ", len(av_avg_loader.dataset))
        current_acc = eval(model, av_avg_loader, writer, epoch)

        print("Len of av_temp_loader: ", len(av_temp_loader.dataset))
        current_acc = eval(model, av_temp_loader, writer, epoch)

        print("Len of comp_loader: ", len(av_comp_loader.dataset))
        current_acc = eval(model, av_comp_loader, writer, epoch)

        print("Len of local_loader: ", len(av_local_loader.dataset))
        current_acc = eval(model, av_local_loader, writer, epoch)

        print("Len of count_loader: ", len(av_count_loader.dataset))
        current_acc = eval(model, av_count_loader, writer, epoch)

        print("Len of exist_loader: ", len(av_exist_loader.dataset))
        current_acc = eval(model, av_exist_loader, writer, epoch)


        print("\n--------------- V --------------- \n")
        print("Len of avg_loader: ", len(v_avg_loader.dataset))
        current_acc = eval(model, v_avg_loader, writer, epoch)

        print("Len of local_loader: ", len(v_local_loader.dataset))
        current_acc = eval(model, v_local_loader, writer, epoch)

        print("Len of count_loader: ", len(v_count_loader.dataset))
        current_acc = eval(model, v_count_loader, writer, epoch)

        print("\n--------------- A --------------- \n")
        print("Len of avg_loader: ", len(a_avg_loader.dataset))
        current_acc = eval(model, a_avg_loader, writer, epoch)

        print("Len of comp_loader: ", len(a_comp_loader.dataset))
        current_acc = eval(model, a_comp_loader, writer, epoch)

        print("Len of count_loader: ", len(a_count_loader.dataset))
        current_acc = eval(model, a_count_loader, writer, epoch)

if __name__ == '__main__':
    main()