import torch 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


import os 
from PIL import Image
import matplotlib.pyplot as plt


class image_dataset(Dataset):
	def __init__(self,input_transform=None,grd_transform = None):
		img_folder_path = 'files/torch7pixel'
		file_list = os.listdir(os.path.join(img_folder_path))
		self.file_list = [os.path.join(img_folder_path,f) for f in file_list if os.path.isfile(os.path.join(img_folder_path,f))]
		print(file_list)
		self.input_transform = input_transform
		self.grd_transform = grd_transform
		
	def __len__(self):
		#return len(self.file_list)
		return 1
	def __getitem__(self,idx):
		img = Image.open(self.file_list[:1][idx])
		grd = Image.open(self.file_list[1:][idx])
		#print(self.file_list[:1][idx],self.file_list[1:][idx])
		if self.input_transform:
			img = self.input_transform(img)
		if self.grd_transform:
			grd = self.grd_transform(grd)
		return img,grd

input_transform = transforms.Compose([
	transforms.ToTensor()
])
grd_transform = transforms.Compose([
	transforms.ToTensor()
])	

myimg_dataset = image_dataset(input_transform = input_transform,grd_transform = grd_transform)
myimg_dataloader = DataLoader(myimg_dataset)
	
class sevenpixels(nn.Module):
	def __init__(self):
		super().__init__()
		"""
		self.layer1_z1 = nn.Conv2d(in_channels=3,out_channels=7,kernel_size=3,bias=True)
		self.layer1_a1 = nn.LeakyReLU(negative_slope=1e-4)
				
		self.layer2_z2 = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=3,bias=True)
		self.layer2_a2 = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3 = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=3,bias=True)
		self.layer3_a3 = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4 = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=3,bias=True)
		self.layer4_a4 = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5 = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True)
		self.layer5_a5 = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6 = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True)
		self.layer6_a6 = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7 = nn.Conv2d(in_channels=7,out_channels=1,kernel_size=3,bias=True)
		self.layer7_a7 = nn.LeakyReLU(negative_slope=1e-4)
		"""
		
		self.layer1_z1_R = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer1_a1_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer1_z1_G = nn.Conv2d(in_channels=1,out_channels=7,kernel_size = 3,bias =True,padding=1)
		self.layer1_a1_G = nn.LeakyReLU(negative_slope=1e-4)		
		
		self.layer1_z1_B = nn.Conv2d(in_channels=1,out_channels=7,kernel_size=3,bias = True,padding=1)
		self.layer1_a1_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer2_a2_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer2_a2_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer2_z2_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer2_a2_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer3_a3_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer3_a3_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer3_z3_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer3_a3_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer4_a4_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer4_a4_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer4_z4_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer4_a4_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer5_a5_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias =True,padding=1)
		self.layer5_a5_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer5_z5_B = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1) 
		self.layer5_a5_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_R = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer6_a6_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_G = nn.Conv2d(in_channels=7,out_channels=7,kernel_size=3,bias=True,padding=1)
		self.layer6_a6_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer6_z6_B = nn.Conv2d(in_channels=7,out_channels =7,kernel_size=3,bias=True,padding=1)
		self.layer6_a6_B = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_R = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size=3,bias=True,padding=1)
		self.layer7_a7_R = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_G = nn.Conv2d(in_channels=7,out_channels =1,kernel_size =3,bias=True,padding=1)
		self.layer7_a7_G = nn.LeakyReLU(negative_slope=1e-4)
		
		self.layer7_z7_B = nn.Conv2d(in_channels=7,out_channels= 1,kernel_size = 3,bias=True,padding=1)
		self.layer7_a7_B = nn.LeakyReLU(negative_slope=1e-4)
		
	
		
	def forward(self,input_image):
		#input_image = [1,3,h,w]
		input_image_r = input_image[:,:1,:,:]
		input_image_g = input_image[:,1:2,:,:]
		input_image_b = input_image[:,2:3,:,:]
		
		layer1_o_r = self.layer1_a1_R(self.layer1_z1_R(input_image_r)) #(batch_size,7,h,w)
		layer1_o_g = self.layer1_a1_G(self.layer1_z1_G(input_image_g))
		layer1_o_b = self.layer1_a1_B(self.layer1_z1_B(input_image_b))
		
		layer2_o_r = self.layer2_a2_R(self.layer2_z2_R(layer1_o_r))
		layer2_o_g = self.layer2_a2_G(self.layer2_z2_G(layer1_o_g))
		layer2_o_b = self.layer2_a2_B(self.layer2_z2_B(layer1_o_b))
		
		layer3_o_r = self.layer3_a3_R(self.layer3_z3_R(layer2_o_r))
		layer3_o_g = self.layer3_a3_G(self.layer3_z3_G(layer2_o_g))
		layer3_o_b = self.layer3_a3_B(self.layer3_z3_B(layer2_o_b))
		
		layer4_o_r = self.layer4_a4_R(self.layer4_z4_R(layer3_o_r))
		layer4_o_g = self.layer4_a4_G(self.layer4_z4_G(layer3_o_g))
		layer4_o_b = self.layer4_a4_B(self.layer4_z4_B(layer3_o_b))
		
		layer5_o_r = self.layer5_a5_R(self.layer5_z5_R(layer4_o_r))
		layer5_o_g = self.layer5_a5_G(self.layer5_z5_G(layer4_o_g))
		layer5_o_b = self.layer5_a5_B(self.layer5_z5_B(layer4_o_b))
		
		layer6_o_r = self.layer6_a6_R(self.layer6_z6_R(layer5_o_r))
		layer6_o_g = self.layer6_a6_G(self.layer6_z6_G(layer5_o_g))
		layer6_o_b = self.layer6_a6_B(self.layer6_z6_B(layer5_o_b))
		
		#print(layer6_o_r.shape) (batch_size,7,H,W)
		
		layer7_o_r = self.layer7_a7_R(self.layer7_z7_R(layer6_o_r))
		layer7_o_g = self.layer7_a7_G(self.layer7_z7_G(layer6_o_g))
		layer7_o_b = self.layer7_a7_B(self.layer7_z7_B(layer6_o_b))
		
		#print(layer7_o_r.shape) #(batch_size,1,H,W)
					
		predicted_o = torch.concatenate((layer7_o_r,layer7_o_g,layer7_o_b),axis = 1)
		#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
		
		return predicted_o
		
		"""
		self.layer1_o = self.layer1_a1(self.layer1_z1(input_image))
		self.layer1_o = torch.where(self.layer1_o>=10,torch.sqrt(90+self.layer1_o),self.layer1_o)   
		print(self.layer1_o.shape,'ehehe')
		
		self.layer2_o = self.layer2_a2(self.layer2_z2(torch.tensor(self.layer1_o)))
		self.layer2_o = torch.where(self.layer2_o>=10,torch.sqrt(90+self.layer2_o),self.layer2_o)

		self.layer3_o = self.layer3_a3(self.layer3_z3(torch.tensor(self.layer2_o)))
		self.layer3_o = torch.where(self.layer3_o>=10,torch.sqrt(90+self.layer3_o),self.layer3_o)
		
		self.layer4_o = self.layer4_a4(self.layer4_z4(torch.tensor(self.layer3_o)))
		self.layer4_o = torch.where(self.layer4_o>=10,torch.sqrt(90+self.layer4_o),self.layer4_o)
		
		self.layer5_o = self.layer5_a5(self.layer5_z5(torch.tensor(self.layer4_o)))
		self.layer5_o = torch.where(self.layer5_o>=10,torch.sqrt(90+self.layer5_o),self.layer5_o)
		
		self.layer6_o = self.layer6_a6(self.layer6_z6(torch.tensor(self.layer5_o)))
		self.layer6_o = torch.where(self.layer6_o>=10,torch.sqrt(90+self.layer6_o),self.layer6_o)
		
		self.layer7_o = self.layer7_a7(self.layer7_z7(torch.tensor(self.layer6_o)))
		self.layer7_o = torch.where(self.layer7_o>=10,torch.sqrt(90+self.layer7_o),self.layer7_o)
		
		return self.layer7_o 
		"""

epoch_count = 0		
		
mypixel_model = sevenpixels()
optimizer = optim.Adam(mypixel_model.parameters(),lr=1e-4)
loss_func = nn.MSELoss()


if os.path.exists('files/torch7pixel/result/torch7pixel_checkpoint.pth'):
	checkpoint = torch.load('files/torch7pixel/result/torch7pixel_checkpoint.pth',map_location=torch.device('cpu'))
	mypixel_model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])		
	epoch_count = checkpoint['epoch_count']
	prev_loss = checkpoint['loss']
	

def update_process():
	xloss_sum = 0
	for batch in myimg_dataloader:
		input_i,output_o = batch
		prediction_o = mypixel_model(input_i)
		#print(prediction_o.shape,output_o.shape)
		xloss = loss_func(prediction_o,output_o)
		optimizer.zero_grad()
		xloss.backward()
		optimizer.step()
		
		print(mypixel_model.layer1_z1_R.weight[0])
		print('####')
		xloss_sum = xloss_sum + xloss
		
	xloss_sum = torch.sum(xloss_sum)
	print(f'epoch_count: {epoch_count}')
	torch.save({'epoch_count':epoch_count,'model_state_dict':mypixel_model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'loss':xloss_sum},'files/torch7pixel/result/torch7pixel_checkpoint.pth')
		
	print(xloss_sum,'kks')
	return xloss_sum
		
		
def main():
	global epoch_count
	prev_loss = 0
	while True:
		current_loss = update_process()
		epoch_count += 1
		if abs(current_loss-prev_loss) > 1e-50:
			prev_loss = current_loss
		else:
			print('Finisheddd')
			break
	print(prev_loss,current_loss)			
		


def show():	
	checkpoint = torch.load('files/torch7pixel/result/torch7pixel_checkpoint.pth',map_location=torch.device('cpu'))
	state_dict = checkpoint['model_state_dict']
	#print(type(state_dict))
	#print(state_dict)
	#print(state_dict[0])
	
	layer1_a1_R = nn.LeakyReLU(negative_slope=1e-4)
	layer1_a1_G = nn.LeakyReLU(negative_slope=1e-4)		
	layer1_a1_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer2_a2_R = nn.LeakyReLU(negative_slope=1e-4)
	layer2_a2_G = nn.LeakyReLU(negative_slope=1e-4)
	layer2_a2_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer3_a3_R = nn.LeakyReLU(negative_slope=1e-4)
	layer3_a3_G = nn.LeakyReLU(negative_slope=1e-4)
	layer3_a3_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer4_a4_R = nn.LeakyReLU(negative_slope=1e-4)
	layer4_a4_G = nn.LeakyReLU(negative_slope=1e-4)
	layer4_a4_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer5_a5_R = nn.LeakyReLU(negative_slope=1e-4)
	layer5_a5_G = nn.LeakyReLU(negative_slope=1e-4)
	layer5_a5_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer6_a6_R = nn.LeakyReLU(negative_slope=1e-4)
	layer6_a6_G = nn.LeakyReLU(negative_slope=1e-4)
	layer6_a6_B = nn.LeakyReLU(negative_slope=1e-4)
	
	layer7_a7_R = nn.LeakyReLU(negative_slope=1e-4)
	layer7_a7_G = nn.LeakyReLU(negative_slope=1e-4)
	layer7_a7_B = nn.LeakyReLU(negative_slope=1e-4)
	
	######
	input_image = 'files/torch7pixel/17.jpg'
	img = Image.open(input_image)
	img = input_transform(img)
	
	print(img.shape,'orginal_image_shape')
	img = torch.stack([img])
	print(img.shape,'orginal_image_shape')
	
	input_image_r = img[:,:1,:,:]
	input_image_g = img[:,1:2,:,:]
	input_image_b = img[:,2:3,:,:]
	
	layer1_o_r = layer1_a1_R(F.conv2d(input=input_image_r,weight=state_dict['layer1_z1_R.weight'],bias= state_dict['layer1_z1_R.bias'],padding=1))
	layer1_o_g = layer1_a1_G(F.conv2d(input=input_image_g,weight=state_dict['layer1_z1_G.weight'],bias= state_dict['layer1_z1_G.bias'],padding=1))
	layer1_o_b = layer1_a1_B(F.conv2d(input=input_image_b,weight=state_dict['layer1_z1_B.weight'],bias= state_dict['layer1_z1_B.bias'],padding=1))
	layer1_o = torch.stack([layer1_o_r,layer1_o_g,layer1_o_b])
	
	layer2_o_r = layer2_a2_R(F.conv2d(input=layer1_o_r,weight=state_dict['layer2_z2_R.weight'],bias= state_dict['layer2_z2_R.bias'],padding=1))
	layer2_o_g = layer2_a2_G(F.conv2d(input=layer1_o_g,weight=state_dict['layer2_z2_G.weight'],bias= state_dict['layer2_z2_G.bias'],padding=1))
	layer2_o_b = layer2_a2_B(F.conv2d(input=layer1_o_b,weight=state_dict['layer2_z2_B.weight'],bias= state_dict['layer2_z2_B.bias'],padding=1))
	layer2_o = torch.stack([layer2_o_r,layer2_o_g,layer2_o_b])
	
	layer3_o_r = layer3_a3_R(F.conv2d(input=layer2_o_r,weight=state_dict['layer3_z3_R.weight'],bias= state_dict['layer3_z3_R.bias'],padding=1))
	layer3_o_g = layer3_a3_G(F.conv2d(input=layer2_o_g,weight=state_dict['layer3_z3_G.weight'],bias= state_dict['layer3_z3_G.bias'],padding=1))
	layer3_o_b = layer3_a3_B(F.conv2d(input=layer2_o_b,weight=state_dict['layer3_z3_B.weight'],bias= state_dict['layer3_z3_B.bias'],padding=1))
	layer3_o = torch.stack([layer3_o_r,layer3_o_g,layer3_o_b])
	
	layer4_o_r = layer4_a4_R(F.conv2d(input=layer3_o_r,weight=state_dict['layer4_z4_R.weight'],bias= state_dict['layer4_z4_R.bias'],padding=1))
	layer4_o_g = layer4_a4_G(F.conv2d(input=layer3_o_g,weight=state_dict['layer4_z4_G.weight'],bias= state_dict['layer4_z4_G.bias'],padding=1))
	layer4_o_b = layer4_a4_B(F.conv2d(input=layer3_o_b,weight=state_dict['layer2_z2_B.weight'],bias= state_dict['layer4_z4_B.bias'],padding=1))
	layer4_o = torch.stack([layer4_o_r,layer4_o_g,layer4_o_b])
	
	layer5_o_r = layer5_a5_R(F.conv2d(input=layer4_o_r,weight=state_dict['layer5_z5_R.weight'],bias= state_dict['layer5_z5_R.bias'],padding=1))
	layer5_o_g = layer5_a5_G(F.conv2d(input=layer4_o_g,weight=state_dict['layer5_z5_G.weight'],bias= state_dict['layer5_z5_G.bias'],padding=1))
	layer5_o_b = layer5_a5_B(F.conv2d(input=layer4_o_b,weight=state_dict['layer5_z5_B.weight'],bias= state_dict['layer5_z5_B.bias'],padding=1))
	layer5_o = torch.stack([layer5_o_r,layer5_o_g,layer5_o_b])
	
	layer6_o_r = layer6_a6_R(F.conv2d(input=layer5_o_r,weight=state_dict['layer6_z6_R.weight'],bias= state_dict['layer6_z6_R.bias'],padding=1))
	layer6_o_g = layer6_a6_G(F.conv2d(input=layer5_o_g,weight=state_dict['layer6_z6_G.weight'],bias= state_dict['layer6_z6_G.bias'],padding=1))
	layer6_o_b = layer6_a6_B(F.conv2d(input=layer5_o_b,weight=state_dict['layer6_z6_B.weight'],bias= state_dict['layer6_z6_B.bias'],padding=1))
	layer6_o = torch.stack([layer6_o_r,layer6_o_g,layer6_o_b])
	
	print(layer6_o_r.shape) #(batch_size,7,H,W)
	
	layer7_o_r = layer7_a7_R(F.conv2d(input=layer6_o_r,weight=state_dict['layer7_z7_R.weight'],bias= state_dict['layer7_z7_R.bias'],padding=1))
	layer7_o_g = layer7_a7_G(F.conv2d(input=layer6_o_g,weight=state_dict['layer7_z7_G.weight'],bias= state_dict['layer7_z7_G.bias'],padding=1))
	layer7_o_b = layer7_a7_B(F.conv2d(input=layer6_o_b,weight=state_dict['layer7_z7_B.weight'],bias= state_dict['layer7_z7_B.bias'],padding=1))
	layer7_o = torch.stack([layer7_o_r,layer7_o_g,layer7_o_b])
	
	print(layer7_o_r.shape) #(batch_size,1,H,W)
				
	predicted_o = torch.concatenate((layer7_o_r,layer7_o_g,layer7_o_b),axis = 1)
	#print(predicted_o.shape,'cxcx kk') (batch_size,3,H,W)
	all_layers = torch.stack([layer1_o,layer2_o,layer3_o,layer4_o,layer5_o,layer6_o])
	
	
	#layer1_0 = r(1,7,h,w) , g(1,7,h,w) , b(1,7,h,w)
	for t in range(6):
		for i in range(7):
			plt.subplot(1,7,i+1)
			print(all_layers[t][0][0][i][:,:].shape,'one_channel')
			
			img_r = torch.unsqueeze(all_layers[t][0][0][i][:,:],axis=-1)
			img_g = torch.unsqueeze(all_layers[t][1][0][i][:,:],axis=-1)
			img_b = torch.unsqueeze(all_layers[t][2][0][i][:,:],axis=-1)
			
			image  = torch.concatenate((img_r,img_g,img_b),axis=2)
			print('image:shape',image.shape)
			plt.imshow(image)
			plt.title(f'layer:{t+1} neuron: {i+1}')
		plt.show()
	
	plt.subplot(1,1,1)
	
	img_r = torch.unsqueeze(layer7_o_r[0][0],axis=-1)
	img_g = torch.unsqueeze(layer7_o_g[0][0],axis=-1)
	img_b = torch.unsqueeze(layer7_o_b[0][0],axis=-1)
	
	img = torch.concatenate((img_r,img_g,img_b),axis=-1)
	plt.imshow(img)
	plt.show()
	
	
	

	
#t = main()
show()








		
		