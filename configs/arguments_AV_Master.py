import os
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')


### ======================== Dataset Configs ==========================

server_path = "/home/data/MUSIC-AVQA/"

parser.add_argument("--audios_feat_dir", type=str, default=os.path.join(server_path, 'vggish'), 
                    help="audio feat dir")
parser.add_argument("--visual_feat_dir", type=str, default=os.path.join(server_path, 'res18'), 
                    help="visual feat dir")
parser.add_argument("--clip_vit_b32_dir", type=str, default=os.path.join(server_path, 'clip_patch-level_vit_b32'), 
                    help="clip_vit-b32_dir")
parser.add_argument("--clip_qst_dir", type=str, default=os.path.join(server_path, 'clip_qst'), 
                    help="clip_qst")
parser.add_argument("--clip_word_dir", type=str, default=os.path.join(server_path, 'clip_word'), 
                    help="clip_word")
parser.add_argument("--frames_dir", type=str, default=os.path.join(server_path, 'avqa-frames-1fps'), 
                    help="video frames dir")
parser.add_argument("--video_res14x14_dir", type=str, default=os.path.join(server_path, 'visual_14x14'), 
                    help="res14x14 dir")
parser.add_argument("--clip_patch_dir", type=str, default=os.path.join(server_path, 'None'), 
                    help="clip_patch_dir")

parser.add_argument("--dataset", type=str, default='music_avqa', 
                    help="dataset")


### ======================== Label Configs ==========================
parser.add_argument("--label_train", type=str, default="./dataset/split_que_id/music_avqa_train.json", 
                    help="train csv file")
parser.add_argument("--label_val", type=str, default="./dataset/split_que_id/music_avqa_val.json", 
                    help="val csv file")
parser.add_argument("--label_test", type=str, default="./dataset/split_que_id/music_avqa_test.json", 
                    help="test csv file")


### ======================== Model Configs ==========================

# ---> TSSM
parser.add_argument("--question_encoder", type=str, default='CLIP', metavar='qe',
                    help="quesiton encoder, CLIP or LSTM")
parser.add_argument("--visual_encoder", type=str, default='CLIP', metavar='ve',
                    help="quesiton encoder, CLIP or ResNet-18 or Swin_V2_L")

# ---> TRSM
parser.add_argument("--spatial_qst_encoder", type=str, default='CLIP', metavar='trqe',
                    help="quesiton encoder, CLIP or LSTM")
parser.add_argument("--spatial_vis_encoder", type=bool, default=True, metavar='sve',
                    help="Spatial regions selector module, Use CLIP")
parser.add_argument("--use_word", type=bool, default=True, metavar='uc',
                    help="word encoder module")

parser.add_argument("--temp_select", type=bool, default=True, metavar='tsm',
                    help="temporal segments selection module")
parser.add_argument("--segs", type=int, default=12, metavar='SEG',
                    help="temporal segment numbers segments")
parser.add_argument("--top_k", type=int, default=2, metavar='TK',
                    help="top K temporal segments")

parser.add_argument("--spat_select", type=bool, default=True, metavar='ssm',
                    help="spatio regions selection module")
parser.add_argument("--top_m", type=int, default=25, metavar='TM',
                    help="top M spatial regions")

parser.add_argument("--a_guided_attn", type=bool, default=True, metavar='ssm',
                    help="audio guided visual attention")


parser.add_argument("--global_local", type=bool, default=True, metavar='glm',
                    help="global local perception module")

parser.add_argument("--temp_grd", type=bool, default=False, metavar='tgm',
                    help="temporal grounding module")

parser.add_argument("--num_layers", type=int, default=1, metavar='num_layers',
                    help="num_layers")


### ======================== Learning Configs ==========================
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=30, metavar='E', 
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', 
                    help='learning rate (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S', 
                    help='random seed (default: 1)')


### ======================== Save Configs ==========================
parser.add_argument( "--checkpoint", type=str, default='AV_Master', 
                    help="save model name")
parser.add_argument("--model_save_dir", type=str, default='', 
                    help="model save dir")
parser.add_argument("--mode", type=str, default='train', 
                    help="with mode to use")


### ======================== Runtime Configs ==========================
parser.add_argument('--log-interval', type=int, default=50, metavar='N', 
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_workers', type=int, default=12, 
                    help='num_workers number')
parser.add_argument('--gpu', type=str, default='1', 
                    help='gpu device number')
