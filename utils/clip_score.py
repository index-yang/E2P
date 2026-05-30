import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from turtle import forward
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# learn_prompt=Prompts().cuda()
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

def get_clip_score_from_feature(tensor,text_features,model):
	score=0
	rate = {}
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2.reshape(1,3,224,224))
		image_features = model.encode_image(image)
		image_nor=image_features.norm(dim=-1, keepdim=True)
		nor= text_features.norm(dim=-1, keepdim=True)
		similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1)
		probs = similarity

		rate[i] = probs[0].cpu().detach().numpy()

		prob = similarity[0][0]
		score = score + prob
	score=score/tensor.shape[0]
	return score,rate


class L_clip_from_feature(nn.Module):
	def __init__(self,model):
		super(L_clip_from_feature,self).__init__()
		# model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#"ViT-B/32"
		# model_path = './prompt-ckpt/best_model_base5.pth'
		# model.load_state_dict(torch.load(model_path, map_location=device))
		# model.to(device)
		# for para in model.parameters():
		# 	para.requires_grad = False
		self.model = model
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, text_features):
		k1,rate = get_clip_score_from_feature(x,text_features,self.model)
		return k1,rate


if __name__ == "main":
	pass