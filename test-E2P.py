import os
import torch
print(torch.cuda.is_available()) 
print(torch.cuda.current_device())
import time, argparse
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from utils.utils import print_args, load_restore_ckpt,tensor_metric,load_restore_ckpt_with_optim

import clip
from model.Prompt import Prompts,TextEncoder
from utils.clip_score import L_clip_from_feature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_model, preprocess = clip.load("ViT-B/32", device = torch.device("cpu"), download_root="./clip_model/")
model_path = './prompt-ckpt/model_iter_12000_klrate.pth'
CLIP_model.load_state_dict(torch.load(model_path, map_location=device))
CLIP_model.to(device)
for para in CLIP_model.parameters():
    para.requires_grad = False

def getTextFeature():
    # return size = types*512
    length_prompt = 5 # Five Type
    #load clip
    prompt_path = './prompt-ckpt/prompt_iter_12000_klrate.pth'
    learn_prompt=Prompts(CLIP_model,prompt_path).cuda()
    learn_prompt =  torch.nn.DataParallel(learn_prompt)
    for name, param in learn_prompt.named_parameters():
        param.requires_grad_(False)
    text_encoder = TextEncoder(CLIP_model)
    embedding_prompt=learn_prompt.module.embedding_prompt
    tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*length_prompt)]])
    text_features = text_encoder(embedding_prompt,tokenized_prompts)
    
    return text_features

def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print('> Model Initialization...')

    text_features = getTextFeature()
    restorer,_,_ = load_restore_ckpt_with_optim(device, freeze_model=True, ckpt_name=args.restore_model_path)

    test(restorer, text_features, device)

transform_resize = transforms.Compose([
        # transforms.Resize([256,256]), # if you need resize
])
def test(restorer, text_features, device):
    L_clip_Feature = L_clip_from_feature(CLIP_model)
    test_input = "./input"

    input_dir = ["haze","water"]

    for i in range(len(input_dir)):
        output = "./output/" + input_dir[i] + "/"
        os.makedirs(output,exist_ok=True)
        file_list =  os.listdir(f'{test_input}/{input_dir[i]}/')
        file_list.sort()
        for j in range(len(file_list)):
            lq = Image.open(f'{test_input}/{input_dir[i]}/{file_list[j]}').convert("RGB")
            restorer.eval()
            with torch.no_grad():
                lq = np.array(lq)
                # lq = (lq - np.min(lq)) / (np.max(lq) - np.min(lq))
                lq = lq / 255.0
                lq_re = torch.Tensor((lq).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                lq_re = transform_resize(lq_re)

                starttime = time.time()
                
                _, rate = L_clip_Feature(lq_re, text_features)
                
                l, _  = text_features.size()
                # print(l)
                print('Completed: ',file_list[j][:-4] +'.png')
                f = torch.zeros(512).to(device)
                for k in range(l):
                    f = f + text_features[k,:] * rate[0][k]
                text_feature2 = f.unsqueeze(0)

                out_1 = restorer(lq_re, text_feature2)
                
                imwrite(out_1, output + file_list[j][:-4] +'.png', range=(0, 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "SaS Running")

    parser.add_argument("--restore-model-path", type=str, default = "./ckpts/E2P_model.tar", help = 'restore model path')

    argspar = parser.parse_args()

    print_args(argspar)

    main(argspar)
