from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        ImagePath = image_filename
        labelPath = label_filename
        ##奶奶的，给的是压缩文件，得用gzip打开
        ImageData = gzip.open(ImagePath,'rb').read()
        labelData = gzip.open(labelPath,'rb').read()
        # read image
        offset = 0
        fmt_header ='>IIII'   # 以大端的方法读取4个unsight int32
        magic_num,num_image,num_rows,num_cols = struct.unpack_from(fmt_header,ImageData,offset)
    
        #print('魔数：{}，图片数：{}，row：{}'.format(magic_num,num_image,num_rows))
        offset += struct.calcsize(fmt_header)
        fmt_image = '>'+str(num_cols*num_rows)+'B'
        image = np.empty((num_image,num_cols*num_rows),np.float32)
        for i in range(num_image):
            im = struct.unpack_from(fmt_image,ImageData,offset)
            image[i] = np.array(im,np.float32)
            offset += struct.calcsize(fmt_image)
        min_val = np.min(image)
        max_val = np.max(image)
        normalize_image = (image-min_val) /(max_val-min_val)
        self.image = normalize_image
        self.img_row = num_rows
        self.img_column = num_cols
        # read label
        offset = 0
        fmt_header ='>II'   # 以大端的方法读取4个unsight int32
        magic_num,num_label = struct.unpack_from(fmt_header,labelData,offset)
        offset += struct.calcsize(fmt_header)
        fmt_label = '>B'
        label = np.empty((num_label),np.uint8)
        for i in range(num_label):
            lb = struct.unpack_from(fmt_label,labelData,offset)
            label[i] = lb[0]
            offset += struct.calcsize(fmt_label)
        self.label = label
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # index 可能是一个list  背下来
        imags = self.image[index]
        lables = self.label[index]
        if(len(imags.shape) == 1):
            #input imags:H*W*C 
            #output H,W,C
            imgs = self.apply_transforms(imags.reshape(self.img_row, self.img_column, 1))
        else:
            # input imags :Batchsize,H*W*C
            # output Ba,H,W,C
            # 这个写的真牛逼，不用管是不是一维的，直接reshape，copilot牛逼
            #imgs = np.vstack([self.apply_transforms(imag.reshape(28, 28, 1)) for imag in imags])
            # 一开始没有后面的reshape，导致batchsize=1的时候会出问题，少一个维度，这就是vstack的问题
            imgs = np.vstack([self.apply_transforms(imag.reshape(28, 28, 1)) for imag in imags]).reshape(-1,28,28,1)
        return (imgs,lables)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.label.shape[0]
        ### END YOUR SOLUTION