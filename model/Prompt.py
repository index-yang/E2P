import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torchvision
import torch.optim
from collections import OrderedDict
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
length_prompt = 16

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,model,initials=None):
        super(Prompts,self).__init__()
        # model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
        # model_path = './prompt-ckpt/best_model_base5.pth'
        # model.load_state_dict(torch.load(model_path, map_location=device))
        # model.to(device)
        
        for para in model.parameters():
            para.requires_grad = False

        print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            # print(text)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path,map_location='cuda' if torch.cuda.is_available() else 'cpu')
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*length_prompt)," ".join(["X"]*length_prompt)]).requires_grad_())).cuda()

    def forward(self,tensor,flag=1):
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features @ (text_features/nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs
        # threshold = 0.3
        # # binary_tensor = torch.where(probs > threshold, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
        # binary_tensor = torch.sigmoid(100 * (probs - threshold))
        # return probs,binary_tensor

if __name__ == "main":
	pass