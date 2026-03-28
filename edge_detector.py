import numpy as np
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time

np.set_printoptions(threshold= np.inf)
#sample_img_array = [[[12,13,16],[0,10,0],[1,1,1]],[[10,110,120],[0,20,0],[2,23,2]]]


#for pooling
def mypooling(img,num_channel):
    img = img.tolist() if str(type(img)) == "<class 'numpy.ndarray'>" else img

    for i,element in enumerate(img,start = 1):
        if num_channel == 3:
            element.insert(0,[0,0,0])
            element.append([0,0,0])
        else:
            element.insert(0,[0])
            element.append([0])           
            
    listxx = []
    for i in range(len(img[0])):
        if num_channel == 3:
            listxx.append([0,0,0])
        else:
            listxx.append([0])
            
    
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
    #print('shapes are',dz.shape,a1.shape)
    for x,x_data in enumerate(dz,start=0):
        #print(f"x is {x}")
        for y,y_data in enumerate(x_data,start=0):
            #print(f"y is {y}")
            
            matrix_sum = matrix_sum + y_data * a1[x:x+3,y:y+3]
            #print('matrix sum' , matrix_sum)
    return matrix_sum


"""
img1= cv2.imread('files/images/test/1.jpg')

b_img = img1[:,:,:1]
g_img = img1[:,:,1:2]
r_img = img1[:,:,2:]

print('b_img',b_img.shape)

vert_filter =  np.array([[[1],[0],[-1]],[[1],[0],[-1]],[[1],[0],[-1]]])
print(vert_filter)
print(vert_filter.shape)
print(vert_filter.ndim)

b_pooled_img = mypooling(b_img,1).tolist()
print('process started : vertical_edge_detector ')
vert = process(b_img = b_pooled_img,filter= vert_filter)


#vert = process(b_img = b_img,filter=vert_filter)
vert = correct_form(vert)
arrayf = vert
cv2.imshow('vert_',vert)
cv2.waitKey(0)
plt.subplot(1,2,1)
plt.imshow(arrayf,cmap='gray')
plt.title('using matplot a1')

plt.subplot(1,2,2)
plt.imshow(b_img,cmap='gray')
plt.title('original')
plt.show()
"""

"""
with open ('files/Pbbimg.txt','w') as f:
    f.write(str(arrayf))

with open ('files/Pblue1.txt','w') as f:
    f.write(str(b_img))
with open ('files/Pvert1.txt','w') as f:
    f.write(str(vert))
"""
#with open ('files/vert_list.txt','w') as f2:
    #f2.write(str(list_y))


 
#with open ('files/Prectified_vert.txt','w') as fr:
    #fr.write(str(arrayf))
                                                                                       
#cv2.imshow('b_img',b_img)
#cv2.imshow('edge',arrayf)
#cv2.waitKey(0)

"""
da = loadmat('files/ground_truth/test/1.mat')
print(type(da))
print(da.keys())

x = da['groundTruth']
print(type(x))
print(x.ndim) #2
print(len(x)) #1
#[[]]

gt_list = x[0]
print(type(gt_list))
print(gt_list.ndim) #1
print(len(gt_list)) #5
#[a,b,c,d,e]


for i,element in enumerate(gt_list,start=1):
    imgg = element['Boundaries'][0][0]
    with open ('files/Pground.txt','w') as f:
        f.write(str(imgg))    
    print('   groudn_truth    ',imgg.shape)
    plt.subplot(1,len(gt_list)+1, i)
    cv2.imshow('ddd',imgg)
    cv2.waitKey(0)
    plt.imshow(imgg,cmap='gray')
    plt.title(f'ground_truth {i}')
    plt.axis('off')

plt.subplot(1,7,7)
plt.imshow(arrayf,cmap='gray')  
plt.title(f'{len(arrayf)},{len(arrayf[0])}') 
plt.axis('off')
plt.show()


print(len(b_img),len(b_img[0]))
print(len(b_pooled_img),len(b_pooled_img[0]))
print(len(arrayf),len(arrayf[0]))   

"""

        
class mcnn:
    def __init__(self):
        self.img_list=[]
        self.grd_list_file = []
        self.grd_list = []
        self.net_loss= 10
      
        self.w = np.array([[[5],[10],[-11]],[[15],[-6],[8]],[[1],[6],[-9]]])
        self.b = 5

        for i in range(1,3):
            self.img_list.append(cv2.imread(f'files/images/test/{i}.jpg'))
            self.grd_list_file.append(f'files/ground_truth/test/{i}.mat')
            grd_file = loadmat(self.grd_list_file[i-1]) 
            self.grd_list.append(grd_file['groundTruth'][0][0]['Boundaries'][0][0])
    
    def main_process(self,w,b):
        self.a1_list = []
        w = self.w
        b = self.b

        for img in self.img_list:
            a0 = img[:,:,:1]
            #print(a0.ndim,'eeee',a0.shape)
            a0 = mypooling(a0,1).tolist()
            #print('ff',a0.ndim,a0.shape)
            z1 = process(a0,w) + b
            #print(z1.ndim,'z1x',z1.shape)
            z1 = correct_form(z1)
            #print(z1.ndim,'xz1',z1.shape)
            a1 = 1/(1+ np.round(np.exp(-z1),4))
            #a1 = z1
            #a1 = np.where(a1>0.5,0.9999,0)
            self.a1_list.append(a1.squeeze())
        #print(self.a1_list[0].ndim,'  h  ',self.a1_list[0].shape)
        return self.a1_list

    
    def mloss(self):
        net_loss = 0
        for i in range(len(self.a1_list)):
            xloss = abs(self.a1_list[i] - self.grd_list[i])
            xloss = xloss if xloss.shape == (481,321) else xloss.T
            #print(xloss.shape,'its xloss shape')
            net_loss = net_loss + xloss
        self.net_loss = (net_loss ** 2)/len(self.a1_list)
        return self.net_loss
    
    def status_check(self):
        x = abs(np.sum(self.net_loss)-np.sum(self.mloss()))
        #status = (x-0.0005)/0.0005
        #print(f'status : {100-status}%',end='\r')
        print('x:',x)
        print('error: ',np.sum(self.net_loss))
        return x
    

    def check(self):
        ccc=0
        while self.status_check() > 0.005:
            if np.sum(self.net_loss) >= 4000:
                self.xslope = 0.0000001
            else:
                self.xslope = 0.0000001    
        
            
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
            plt.imsave(f'files/test_result/training_two_again/BCE{ccc}.jpg',self.a1_list[0],cmap='gray')
            ccc = ccc+1
            
           
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
                current_img_a0_pooled = mypooling(current_original_img[:,:,:1],1)
                #print('current_img_a0_pooled shape is ',current_img_a0_pooled.shape)
                """
                da1 = current_img_a1 - (current_img_a1 ** 2)
                #print(np.sum(da1),'   da1 sum  ',' da1 shape is ',da1.shape)
                dz1 = (current_img_a1 - current_grd) * da1 
                #print(np.sum(dz1),' dz1 sum  ','and dz1 shape is ',dz1.shape)
                dw1 = myconv_process(dz1,current_img_a0_pooled)
                #print(np.sum(dw1),'dw1 sum','and dw1 shape is ',dw1.shape)
                db1 = dz1 if dz1.shape == (481,321) else dz1.T
                """
                #above for mse loss
                #downside for bce loss
                dz1 = current_img_a1 - current_grd
                dw1 = myconv_process(dz1,current_img_a0_pooled)
                db1 = dz1 if dz1.shape == (481,321) else dz1.T
                

                dw_net = dw1 + dw_net
                db_net = db1 + db_net
            #print('   dw_net    ') 
            #print('slope is ',self.xslope)
            self.w = self.w - self.xslope * dw_net
            self.b = self.b - self.xslope * np.sum(db_net)
            self.main_process(self.w,self.b)
        
        print('completed and  final values of w and b are ')
        print(self.w)
        print(self.b)
        print('error is ',np.sum(self.net_loss))
        
        for i in range(len(self.a1_list)):
            x = (self.a1_list[i] * 255).astype(np.uint8)
            cv2.imwrite(f'files/test_result/trained_two_again{i+1}.jpg',x)
        with open(f'files/test_result/trainingtwo_againFINAL.txt','w') as xfile:
            xfile.write(f'w is {self.w} and b is {self.b}')
            xfile.close()
               
        

        
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
        


t1 = time.time()                    
c1 = mcnn()
c1.main_process(5,6)
c1.check()
t2 = time.time()
print(f'time taken {t2-t1}')
        

        


#grd_truth = gt_list[0]['Boundaries'][0][0]
















      
