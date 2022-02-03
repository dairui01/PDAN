import torch
import torch.nn as nn


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x

class WeightedPermuteMLP(nn.Module):
	def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()

		self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
		self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
		self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

		self.reweight = Mlp(dim, dim // 4, dim *4)
		
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)



	def forward(self, x):
		x = x.unsqueeze(0)		
		B, H, W, C = x.shape

		
		h = self.mlp_h(x)

		w = self.mlp_w(x)

		c = self.mlp_c(x)
		
		a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
		a = self.reweight(a).reshape(B, C, 4).permute(2, 0, 1).softmax(dim=0)

		y = h * a[0] + w * a[1] + c * a[2]

		y = self.proj(y)
		y = self.proj_drop(y)
		x = y + x

		return x.squeeze(0)
