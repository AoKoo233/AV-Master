import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify
import time
import random



def TransformImage(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    transform_list.append(transforms.Resize([256,256]))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_tensor = trans(img)
    
    return frame_tensor

def TransformImage_Resize(img):

    transform_list = []
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    transform_list.append(transforms.Resize([256,256]))
    # transform_list.append(transforms.ToTensor())
    # transform_list.append(transforms.Normalize(mean, std))
    trans = transforms.Compose(transform_list)
    frame_org = trans(img)
    
    return frame_org


def load_frame_info(img_path):

    img = Image.open(img_path).convert('RGB')
    # img2 = TransformImage_Resize(img)   # visualization
    frame_tensor = TransformImage(img)

    # return img2, frame_tensor    # visualization
    return frame_tensor


def image_info(frame_path):

    img_list = os.listdir(frame_path)
    img_list.sort()

    select_img = []
    
    for frame_idx in range(len(img_list)):
        if frame_idx < 60:
            video_frames_path = os.path.join(frame_path, str(frame_idx+1).zfill(6)+".jpg")
            frame_tensor_info = load_frame_info(video_frames_path)
            select_img.append(frame_tensor_info.cpu().numpy())


    select_img = np.array(select_img)

    # return org_img, select_img
    return select_img



def ids_to_multinomial(id, categories):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    return id_to_idx[id]

def get_random_index(n, idx):
    if n <= 1:
        raise SystemError

    # 生成随机索引，确保不等于 idx
    random_index = idx
    while random_index == idx:
        random_index = random.randint(0, n - 1)

    return random_index


class AVQA_dataset(Dataset):

    def __init__(self, args, label, audios_feat_dir, visual_feat_dir, 
                       clip_vit_b32_dir, clip_qst_dir, clip_word_dir,
                       transform=None, mode_flag='train'):

        self.args = args

        samples = json.load(open('./dataset/musci_avqa/music_avqa_train.json', 'r'))


        # Question
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])
        # ques_vocab.append('fifth')

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14    # question length

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir

        self.clip_vit_b32_dir = clip_vit_b32_dir
        self.clip_qst_dir = clip_qst_dir
        self.clip_word_dir = clip_word_dir

        self.transform = transform


    def __len__(self):
        return len(self.samples)

    def get_lstm_embeddings(self, question_input, sample):

        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')

        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        return ques

    def get_frames_spatial(self, video_name):
        
        frames_path = os.path.join(self.frames_dir, video_name)
        frames_spatial = image_info(frames_path)    # [T, 3, 224, 224]

        return frames_spatial

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']

        audios_feat = np.load(os.path.join(self.audios_feat_dir, name + '.npy'))

        
        if self.args.question_encoder == "CLIP":
            question_feat = np.load(os.path.join(self.clip_qst_dir, str(question_id) + '.npy')) 
        else:
            question = sample['question_content']
            question_feat = self.get_lstm_embeddings(question, sample)
            raise SystemError

        if self.args.visual_encoder == "CLIP":
            visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '.npy'))          
            #visual_feat = visual_CLIP_feat[:60, 0, :]
            visual_feat = visual_CLIP_feat[:60, :]
        elif self.args.visual_encoder == "Swin_V2_L":
            visual_feat = np.load(os.path.join(self.visual_feat_dir, name + '.npy'))

        if self.args.spatial_vis_encoder:
            visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '.npy'))
            #patch_feat = visual_CLIP_feat[:60, 1:, :]

            patch_feat = visual_CLIP_feat[:60, :]
        else:
            patch_feat = np.zeros((1, 1), dtype=float)

        if self.args.use_word:
            word_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '.npy'))
        else:
            word_feat = np.zeros((1, 1), dtype=float)


        neg_idx = get_random_index(len(self.samples), idx)
        neg_sample = self.samples[neg_idx]
        neg_name = neg_sample['video_id']
        neg_visual_feat = np.load(os.path.join(self.clip_vit_b32_dir, neg_name + '.npy'))[:60, :]
        neg_audios_feat = np.load(os.path.join(self.audios_feat_dir, neg_name + '.npy'))  
            
        ### answer
        answer = sample['anser']
        answer_label = ids_to_multinomial(answer, self.ans_vocab)
        answer_label = torch.from_numpy(np.array(answer_label)).long()

        sample = {'video_name': name,
                  'audios_feat': audios_feat, 
                  'visual_feat': visual_feat,
                  'patch_feat': patch_feat,
                  'neg_visual_feat' : neg_visual_feat,
                  'neg_audios_feat' : neg_audios_feat,
                  'question': question_feat,
                  'qst_word': word_feat,
                  'answer_label': answer_label, 
                  'question_id': question_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):

        video_name = sample['video_name']
        audios_feat = sample['audios_feat']
        visual_feat = sample['visual_feat']
        patch_feat = sample['patch_feat']
        question = sample['question']
        qst_word = sample['qst_word']
        answer_label = sample['answer_label']
        question_id = sample['question_id']

        neg_visual_feat = sample['neg_visual_feat']
        neg_audios_feat = sample['neg_audios_feat']

        return {'video_name': video_name, 
                'audios_feat': torch.from_numpy(audios_feat),
                'visual_feat': torch.from_numpy(visual_feat),
                'patch_feat': torch.from_numpy(patch_feat),
                'neg_visual_feat' : torch.from_numpy(neg_visual_feat),
                'neg_audios_feat' : torch.from_numpy(neg_audios_feat),
                'question': sample['question'],
                'qst_word': sample['qst_word'],
                'answer_label': answer_label,
                'question_id':question_id}
