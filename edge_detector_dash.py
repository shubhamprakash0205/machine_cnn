import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import os

np.set_printoptions(threshold= np.inf)
#sample_img_array = [[[12,13,16],[0,10,0],[1,1,1]],[[10,110,120],[0,20,0],[2,23,2]]]


#for padding
def mypadding(img,num_channel):
	img = img.tolist() if str(type(img)) == "<class 'numpy.ndarray'>" else img
	element_list=[]
	for i in range(num_channel):
		element_list.append(0)
        
	for i,element in enumerate(img,start = 1):
		element.insert(0,element_list)
		element.append(element_list)          
            
	listxx = []
	for i in range(len(img[0])):
		listxx.append(element_list)
                    
	img.insert(0,listxx)
	img.append(listxx)
    
	return np.array(img)



#convulation
def process(b_img,filter):
	list_y = []
	for y,y_row in enumerate(b_img[:len(b_img)-2]):
		list_x = []
		for x,x_row in enumerate(y_row[:len(y_row)-2]):
            
			yx = b_img[y][x:x+3] * filter[:1]
			yx1 = b_img[y+1][x:x+3] * filter[1:2]
			yx2 = b_img[y+2][x:x+3] * filter[2:3]
			sum1 = np.sum(yx) + np.sum(yx1) + np.sum(yx2)
			list_x.append([sum1])
        
		list_y.append([list_x])
	return np.array(list_y)

def correct_form(vert):
	vert_max = np.max(vert)
	vert_min = np.min(vert)
    
	#print(vert_max)
	#print(vert_min)
	
	listx = []
	for element in vert:
		listx.append(element[0].tolist())
	arrayf = np.array(listx)
	#arrayf = arrayf/(255*3*2)
	#arrayf = arrayf + 0.5
	#arrayf = arrayf/(vert_max-vert_min)
	# arrayf = arrayf + (0 - (vert_min/(vert_max-vert_min)))
    
	#print(np.max(arrayf),'after correction')
	#print(np.min(arrayf),'after correction')
	return arrayf


#coorelation
def myconv_process(dz,a1):
    
	matrix_sum = 0
	#print('conv_process started')
	print('shapes are',dz.shape,a1.shape)
	for x,x_data in enumerate(dz,start=0):
		#print(f"x is {x}")
		for y,y_data in enumerate(x_data,start=0):
			#print(f"y is {y}")
            
			matrix_sum = matrix_sum + y_data * a1[x:x+3,y:y+3]
			#print('matrix sum' , matrix_sum)
	return matrix_sum

        
class mcnn:
	def __init__(self):
		self.img_list=[]
		self.grd_list_file = []
		self.grd_list = []
		self.net_loss= 10
		self.countt = np.array([1600])
		self.prev_a1 = 5

		if os.path.exists('files/test_result/mycheckpoint.npz'):
			checkpoint = np.load('files/test_result/mycheckpoint.npz')
			self.w = checkpoint['w']
			self.b = checkpoint['b']
			self.countt = checkpoint['countt']
		if True:
			self.w = np.array([[[-144.08774935],[-139.02410336],[-158.11474643]],[[-134.54368463],[-155.42413673],[-139.98204685]],[[-145.72833147],[-140.29215656],[-153.4557941]]])
			self.w = np.array([[[5],[10],[-11]],[[15],[-6],[8]],[[1],[6],[-9]]])
			self.b = 5
			#self.w = np.array([[[-1922.88137887],[-1899.84638767],[-1939.91353492]],[[-1939.82849701],[-1898.93319669],[-1917.42170626]],[[-1929.1230905 ],[-1858.62894495],[-1880.96025253]]]) #white one
			#self.b = -8.606837894589951
			self.countt = np.array([900])
			
		for i in range(1,2):
			self.img_list.append(cv2.imread(f'files/images/test/{i}.jpg'))
			self.grd_list_file.append(f'files/ground_truth/test/{i}.mat')
			grd_file = loadmat(self.grd_list_file[i-1])
			for i in range(5): 
				el = grd_file['groundTruth'][0]
				self.grd_list.append(el[i]['Boundaries'][0][0])
		self.grd_list = np.sum(self.grd_list,axis=0,keepdims=True) / 5    
	
	def main_process(self,w,b):
		print()
		print('inside maiN_proceCESS')
		
		if True:
			try:
				self.prev_a1 = self.a1_list[0]
				print('prev_a1_updated')
			except:
				print('Not updated')

		self.a1_list = []
		w = self.w
		b = self.b
		print("w: ",w,"b: ",b)
				
		0-255	
		for img in self.img_list:
			a0 = img[:,:,:1]
			a0 = a0/255 #fucking problem for days #shouldnt be removed
			#print(a0.ndim,'eeee',a0.shape)
			a0 = mypadding(a0,1).tolist()
			#print('ff',a0.ndim,a0.shape)
			z1 = process(a0,w) + b
			#print(z1.ndim,'z1x',z1.shape)
			z1 = correct_form(z1)
			#print(z1.ndim,'xz1',z1.shape)
			#a1 = 1/(1+ np.round(np.exp(-z1),4))
			a1 = 1/(1+ np.exp(-z1))
			
			#a1 = z1
			#a1 = np.where(a1>0.5,0.9999,0)
			self.a1_list.append(a1.squeeze())
		#print(self.a1_list[0].ndim,'  h  ',self.a1_list[0].shape)
		
		if True:
			try:
				if np.array_equal(self.prev_a1,self.a1_list[0]):
					print('FUCKKCKK')
				else:
					print('not so fuck')
					print(np.sum(self.prev_a1),np.sum(self.a1_list[0]))
					print(np.sum(self.prev_a1-self.a1_list[0]))
					print('not so fuck....')
			except Exception as e:
				print(str(e))
			
		
		return self.a1_list

    
	def mloss(self):
		print()
		print('inside MLOSS')
		"""
		#MSE loss:
		net_loss = 0
		for i in range(len(self.a1_list)):
			xloss = abs(self.a1_list[i] - self.grd_list[i])
			xloss = xloss if xloss.shape == (481,321) else xloss.T
			#print(xloss.shape,'its xloss shape')
			net_loss = net_loss + xloss
		self.net_loss = (net_loss ** 2)/len(self.a1_list)
		return self.net_loss
		"""
		
		#BCE loss:
		net_loss = 0
		for i in range(len(self.a1_list)): 
			processing_grd = np.where(self.grd_list[i]==0,0.000001,self.grd_list[i])
			processing_grd = np.where(processing_grd==1,0.999999,processing_grd)
			processing_a1 = np.where(self.a1_list[i]==0,0.000001,self.a1_list[i])
			processing_a1 = np.where(processing_a1==1,0.999999,processing_a1)
			xloss = (-processing_grd * np.log(processing_a1)) - ((1-processing_grd) * np.log(1-processing_a1))                       	
			xloss = xloss if xloss.shape == (481,321) else xloss.T
			print('xloss: ',np.sum(xloss))
			net_loss = net_loss + xloss
		print('self.net_loss: ',np.sum(self.net_loss))
		self.net_loss = net_loss
		return self.net_loss
    		
	def status_check(self):
		print()
		print('inside STATUS CHECK')
		if self.countt % 1 == np.array([0]):
			ff = open('files/test_result/FINAL.txt','a')
			gg = open('files/test_result/FINAL_ONE.txt','w')
			ff.write(f'self.w is {self.w} and b is {self.b} .Also error is {np.sum(self.net_loss)} , count:{self.countt}')
			gg.write(f'w: {self.w} and b: {self.b}')
			np.savez('files/test_result/mycheckpoint.npz',w = self.w,b = self.b,countt = self.countt)
			
			
		print('self.countt: ',self.countt)
		
		x = abs(np.sum(self.net_loss)-np.sum(self.mloss()))
		#status = (x-0.0005)/0.0005
		#print(f'status : {100-status}%',end='\r')
		print('x:',x)
		print('error: ',np.sum(self.net_loss))
		self.countt = self.countt+1
		return x
    
    
	
	
		
	def check(self):
		while self.status_check() > 0.00000000000000005:
			print()
			print('INSIDE CHECK')
			if np.sum(self.net_loss) >= 4000:
				self.xslope = 0.0001 #0.0001 
			else:
				self.xslope = 0.0001   

			
			
			
			plt.subplot(1,3,1)
			plt.imshow(self.img_list[0][:,:,:1],cmap='gray')
			plt.title('original image')
		
			plt.subplot(1,3,2)
			plt.imshow(self.a1_list[0],cmap='gray')
			plt.title('a1')
	
			plt.subplot(1,3,3)
			plt.imshow(self.grd_list[0],cmap='gray')
			plt.title('grd')
			#plt.show()
			plt.imsave(f'files/test_result/training_two_again/BCExxx{self.countt}.jpg',self.a1_list[0],cmap='gray')
			#plt.imsave(f'files/test_result/training_two_again/BGRDxxx{self.countt}.jpg',self.grd_list[0],cmap='gray')
			

					
		   
			#print('error is ',np.sum(self.net_loss))
			#print('self.w is ',self.w,'and self.b is ',self.b)
			dw_net = 0
			db_net = 0
			for i in range(len(self.a1_list)):
				current_img_a1 = self.a1_list[i]
				#print('current_img_a1 shape is ',current_img_a1.shape) 
				current_grd = self.grd_list[i]
				#print('current grd shape is ',current_grd.shape)
				current_original_img = self.img_list[i]
				#print('current_original_img shape is ',current_original_img.shape) 
				current_img_a0_padded = mypadding(current_original_img[:,:,:1],1)
				#print('current_img_a0_padded shape is ',current_img_a0_padded.shape)
				"""
				da1 = current_img_a1 - (current_img_a1 ** 2)
				#print(np.sum(da1),'   da1 sum  ',' da1 shape is ',da1.shape)
				dz1 = (current_img_a1 - current_grd) * da1 
				#print(np.sum(dz1),' dz1 sum  ','and dz1 shape is ',dz1.shape)
				dw1 = myconv_process(dz1,current_img_a0_padded)
				#print(np.sum(dw1),'dw1 sum','and dw1 shape is ',dw1.shape)
				db1 = dz1 if dz1.shape == (481,321) else dz1.T
				"""
				#above for mse loss
				#downside for bce loss
				
				dz1 = current_img_a1 - current_grd
				print(np.sum(dz1),'hhrf')
				
				dw1 = myconv_process(dz1,current_img_a0_padded)
				#print('dw1__',dw1)
				db1 = dz1 if dz1.shape == (481,321) else dz1.T
				
	
				dw_net = dw1 + dw_net
				db_net = db1 + db_net
			print('   dw_net    ',dw_net) 
			#print('slope is ',self.xslope)
			print('self.w :',self.w)
			self.w = self.w - self.xslope * dw_net
			self.b = self.b - self.xslope * np.sum(db_net)
			print('self.w after update :',self.w)
			self.main_process(self.w,self.b)
		
		#print('completed and  final values of w and b are ')
		#print(self.w)
		#print(self.b)
		#print('error is ',np.sum(self.net_loss))
		
		with open(f'files/test_result/trainingtwo_againFINAL.txt','w') as xfile:
			xfile.write(f'w is {self.w} and b is {self.b}')
			xfile.close()
		   
		

		"""
		plt.subplot(1,3,2)
		plt.imshow(self.a1_list[0],cmap='gray')
		plt.title('a1_final')
		
		plt.subplot(1,3,3)
		plt.imshow(self.grd_list[0],cmap='gray')
		plt.title('ground_truth')
		
		plt.subplot(1,3,1)
		plt.imshow(self.img_list[0][:,:,:1],cmap='gray')
		plt.title('reAL IMG')
		plt.show()

       
		plt.subplot(1,3,2)
		plt.imshow(self.a1_list[1],cmap='gray')
		plt.title('a1_final')
		
		plt.subplot(1,3,3)
		plt.imshow(self.grd_list[1],cmap='gray')
		plt.title('ground_truth')
		
		plt.subplot(1,3,1)
		plt.imshow(self.img_list[1][:,:,:1],cmap='gray')
		plt.title('reAL IMG')
		plt.show()
		"""
		"""
		plt.subplot(1,9,5)
		plt.imshow(self.a1_list[1],cmap='gray')
		plt.title('a1_final')
		
		plt.subplot(1,9,6)
		plt.imshow(self.grd_list[1],cmap='gray')
		plt.title('ground_truth')
		
		plt.subplot(1,9,4)
		plt.imshow(self.img_list[1][:,:,:1],cmap='gray')
		plt.title('reAL IMG')

		plt.subplot(1,9,8)
		plt.imshow(self.a1_list[2],cmap='gray')
		plt.title('a1_final')
		
		plt.subplot(1,9,9)
		plt.imshow(self.grd_list[2],cmap='gray')
		plt.title('ground_truth')
		
		plt.subplot(1,9,7)
		plt.imshow(self.img_list[2][:,:,:1],cmap='gray')
		plt.title('reAL IMG')
		"""
		
	def cross_check(self):
		self.img_list = []
		self.grd_list_file = []
		self.grd_list = []
		for i in range(10,20):
			self.img_list.append(cv2.imread(f'files/images/test/{i}.jpg'))
			self.grd_list_file.append(f'files/ground_truth/test/{i}.mat')
			grd_file = loadmat(self.grd_list_file[i-10]) 
			self.grd_list.append(grd_file['groundTruth'][0][0]['Boundaries'][0][0])
	
		self.main_process(5,6)
		
		for i in range(10):
			plt.subplot(1,3,1)
			plt.imshow(self.img_list[i][:,:,:1],cmap='gray')
			plt.title(i)

			plt.subplot(1,3,2)
			plt.imshow(self.a1_list[i],cmap='gray')
			plt.title(i)
			
			plt.subplot(1,3,3)
			plt.imshow(self.grd_list[i],cmap='gray')
			plt.title(i)
			
			plt.show()
			




#t1 = time.time()				
c1 = mcnn()
c1.main_process(5,6)
#c1.status_check()
c1.check()
#t2 = time.time()
#print(f'time taken {t2-t1}')
#c1.cross_check() 
#print(c1.w,c1.b)    

        


#grd_truth = gt_list[0]['Boundaries'][0][0]
















      

