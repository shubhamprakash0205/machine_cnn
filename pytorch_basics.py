import torch

x = torch.tensor([[1,2,3],[34.5,6,2]])
print(x)
y = torch.randn(1,10)
print(y)


if torch.cuda.is_available():
	print('raban')
else:
	print('ram') 

z = torch.randn((1,10),requires_grad=True)
print(z)
print(z.grad_fn)
y = torch.sqrt(z)
print(y)
y= torch.tensor(y,requires_grad=True)
print(y.grad_fn)
print(y)

print()
x1 = torch.tensor(2.0,requires_grad = True)
y1 = torch.tensor(3.6,requires_grad = True)
print(x1)
print(y1)
z1 = (x1 * y1)
xloss = z1 - 6
print(xloss.grad_fn)
xloss.backward()
#print('dL/dL=',xloss.grad)
#print('dL/z1=',z1.grad)
print('dL/x1=',x1.grad)
print('dL/y1=',y1.grad)

from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import os
class xxdataset(Dataset):
	def __init__(self,transform=None):
		self.data = [(i,i**2) for i in range(20) if i % 2 == 0]
		self.transform = transform
		
	def __len__(self):
		return(len(self.data))
	def __getitem__(self,idx):
		inp = torch.tensor(self.data[idx][0])
		out = torch.tensor(self.data[idx][1])

		if self.transform:
			inp = self.transform(inp)
			out = self.transform(out)
			
		return inp,out

xxtransform = transforms.Compose([
	#transforms.ToTensor()
	])
#if you keep something like transform.totensor in above xxtransform , it will give you 
#error as these things can be just applied PIL images objects or numpy arrays 

cxdataset = xxdataset(transform=xxtransform)
xxloader = DataLoader(cxdataset,batch_size=3,shuffle = True)
for data in xxloader:
	print(data) # it will return batches of [array of 3inputs,array of corrosponding 3outputs of that 3inputs] 


class xdataset(Dataset):
	def __init__(self,image_folder_path,grd_folder_path,input_transform=None,output_transform=None):
		self.image_files_list = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path,f))]
		self.image_files_list = [c for c in self.image_files_list if os.path.splitext(c)[1] == '.jpg']
		self.image_files_list = sorted(self.image_files_list)
		
		self.grd_files_list = [f for f in os.listdir(grd_folder_path) if os.path.isfile(os.path.join(grd_folder_path,f))]
		self.grd_files_list = [c for c in self.grd_files_list if os.path.splitext(c)[1] == '.mat']
		self.grd_files_list = sorted(self.grd_files_list)
		
		self.input_transform = input_transform
		self.output_transform = output_transform
		
	def __len__(self):
		return len(self.image_files_list)
	
	def __getitem__(self,idx):
		
		image = Image.open(self.image_files_list[idx]).convert('L') #grayscale mode
		grd = Image.open(self.grd_files_list[idx]).convert('L')
		
		if self.input_transform:
			image = self.input_transform(image)
		if self.output_transform:
			grd = self.output_transform
		
		return image,grd
	
input_transform = transforms.Compose([
	transforms.ToTensor()
	])

output_transform = transforms.Compose([
	transforms.ToTensor()
	])

mydataset = xdataset('files/images/train','files/ground_truth/train',input_transform,output_transform)
loader = DataLoader(mydataset,batch_size=2,shuffle = True)
print(len(loader))


import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print('pandas_version:::',pd.__version__)
print()
df = pd.read_csv('files/basics/housePrice.csv')
#print(df.head(10))
#print(df.sample(3))
"""
df2 = pd.DataFrame({'name':['Ram','Shyam','Ravan'],'Age':[12,13,18],'Address':['Panvel','Nonvel','Soundwell']})
print(df2)
print(df2.index)
print(df2.loc[1])
print(df2.iloc[2])
idx = pd.Index(["a","b","c","d"], name="letters")
print(df2.loc[0])
print(idx)
#print(df2.loc)
#print(df2.iloc)

"""
df3 = pd.read_csv('files/basics/housePrice.csv',usecols = ['Area','Room','Parking','Warehouse','Elevator','Address','Price(USD)'])
print(df3.head(8))
#print(df3['Address'])
print()
print(df3['Address'].nunique())
print(df3['Address'].value_counts())
print(type(df3['Address']))
print(df3['Address'][3])


print()
x= df3.drop('Address',axis=1)
print(df3.head(2))
print()
print(x.head(2))
print()
xx = x.to_numpy()
print(len(xx))
print(xx[1],type(xx[1]),xx.shape)




