import numpy as np
import cv2
import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
from helper.align import kestrel_get_similar_matrix

cv2.ocl.setUseOpenCL(False)

class FaceAugmentation(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        w, h = img.size
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        img = img.resize((self.final_size, self.final_size), box=rect)

        ## to BGR
        img = np.array(img)
        img = img[:,:,[2,1,0]]

        return img

class FaceAugmentationCV2(object):
    def __init__(self, crop_size, final_size, crop_center_x_offset, \
                crop_center_y_offset, scale_aug, trans_aug, flip=-1, \
                mask_aug=0, half=0):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.crop_center_x_offset = crop_center_x_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        self.mask_aug = mask_aug
        self.half = half

    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = (w/2+self.crop_center_x_offset)*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]

        img = cv2.resize(img, (self.final_size, self.final_size))

        if self.half == 1:

            img[:self.final_size // 2, :] = 0

        elif self.half == -1:

            img[self.final_size // 2 :, :] = 0

        #self.mask_aug = 1
        if self.mask_aug > 0:

            seed = np.random.rand()
            aug = self.final_size // 2 * min(np.random.rand() + 0.1, 1)
            if seed < self.mask_aug / 2:
                img[:int(self.final_size//2-aug), :] = 0
            elif seed < self.mask_aug:
                img[int(self.final_size//2+aug):, :] = 0

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        ## to BGR
        #img = np.array(img)
        #img = img[:,:,[2,1,0]]

        return img

class FaceAugmentationCV2Mask(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug, mask_spec,
                total_ratio,sunglass,sunglass_ratio,mask,mask_ratio,hat,hat_ratio,flip=-1,mask_type='random'):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        #####add mask_spec by wl
        self.mask_spec = mask_spec
        self.total_ratio = total_ratio
        self.sunglass = sunglass
        self.sunglass_ratio = sunglass_ratio
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.hat = hat
        self.hat_ratio = hat_ratio

    def __call__(self, img):
        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]
        img = cv2.resize(img, (self.final_size, self.final_size))

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        ## to BGR
        #img = np.array(img)
        #img = img[:,:,[2,1,0]]
        if self.mask_spec == True:
            LEx = 70.7
            LEy = 113.0
            REx = 108.23
            REy = 113.0
            Mx = 89.43
            My = 153.51
            LEx_mod = (LEx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            LEy_mod = (LEy - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)
            REx_mod = (REx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            REy_mod = (REy - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)
            Mx_mod = (Mx - (ct_x - crop_aug_w/2))*(self.final_size/crop_aug_w)
            My_mod = (My - (ct_y - crop_aug_h/2))*(self.final_size/crop_aug_h)

            print_flag = False
            totaltemprand = np.random.rand()
            if totaltemprand < self.total_ratio:
                temprand = np.random.rand()
                if self.sunglass > 0 and temprand < self.sunglass_ratio:
                    # radious = 30 + np.random.rand()*15
                    radious = self.final_size/2.0/4.0 + np.random.rand()*(self.final_size/2.0/4.0/2.0)
                    cv2.circle(img,(int(LEx_mod),int(LEy_mod)),int(radious),(0,0,0),-1)
                    cv2.circle(img,(int(REx_mod),int(REy_mod)),int(radious),(0,0,0),-1)
                    if print_flag:
                        print ("sunglass")
                        cv2.imwrite("sunglass3.jpg",img)

                elif self.hat>0 and (temprand - self.sunglass_ratio) < self.hat_ratio:
                    hat_w = (REx_mod - LEx_mod)*2.4
                    hat_h = (LEy_mod)*(0.7+(1.2-0.7)*np.random.rand())
                    hat_l = max(int((REx_mod + LEx_mod)/2 - hat_w/2),1)
                    hat_t = 1
                    hat_r = min(int(hat_l + hat_w),(self.final_size - 1))
                    hat_b = min(int(hat_t + hat_h),(self.final_size - 1))
                    for i in range(hat_t,hat_b+1):
                        for j in range(hat_l,hat_r+1):
                            img[i,j][0] = np.random.randint(0,256)
                            img[i,j][1] = np.random.randint(0,256)
                            img[i,j][2] = np.random.randint(0,256)
                    if print_flag:
                        print("hat")
                        cv2.imwrite("hat3.jpg",img)

                elif self.mask > 0 :
                    mask_w = (REx_mod - LEx_mod)*2.4
                    mask_h = (self.final_size - My_mod)*(1.6+(2-1.6)*np.random.rand())
                    mask_l = max(int((REx_mod + LEx_mod)/2 - mask_w/2),1)
                    mask_r = min(int(mask_l+mask_w),(self.final_size - 1))
                    mask_t = max(int(self.final_size  - mask_h),1)
                    mask_b = self.final_size - 1
                    for i in range(mask_t,mask_b+1):
                        for j in range(mask_l,mask_r + 1):
                            img[i,j][0] = np.random.randint(0,256)
                            img[i,j][1] = np.random.randint(0,256)
                            img[i,j][2] = np.random.randint(0,256)
                    if print_flag:
                        print("mask")
                        cv2.imwrite("mask3.jpg",img)
            else:
                if print_flag:
                    print("Nothing")
        return img


class FaceAugmentationCV2Template(object):
    def __init__(self, crop_size, final_size, crop_center_y_offset, scale_aug, trans_aug, mask_spec,
                total_ratio,sunglass,sunglass_ratio,mask,mask_ratio,hat,hat_ratio,flip=-1,mask_type='template'):
        self.crop_size = crop_size
        self.final_size = final_size
        self.crop_center_y_offset = crop_center_y_offset
        self.scale_aug = scale_aug
        self.trans_aug = trans_aug
        self.flip = flip
        #####add mask_spec by wl
        self.mask_spec = mask_spec
        self.total_ratio = total_ratio
        self.sunglass = sunglass
        self.sunglass_ratio = sunglass_ratio
        self.mask = mask
        self.mask_ratio = mask_ratio
        self.hat = hat
        self.hat_ratio = hat_ratio

    def __call__(self, img):
        is_sunglass = False
        is_hat = False
        is_mask = False
        print_flag = False

        if self.mask_spec == True:
            LEx = 70.7
            LEy = 113.0
            REx = 108.23
            REy = 113.0
            Mx = 89.43
            My = 153.51

            totaltemprand = np.random.rand()
            if totaltemprand < self.total_ratio:
                temprand = np.random.rand()

                workpath = os.path.abspath('.')

                if self.sunglass > 0.0 and temprand <= self.sunglass_ratio:
                    radious = 15 + np.random.rand()*15
                    #radious = self.final_size/2.0/4.0 + np.random.rand()*(self.final_size/2.0/4.0/2.0)
                    cv2.circle(img,(int(LEx),int(LEy)),int(radious),(0,0,0),-1)
                    cv2.circle(img,(int(REx),int(REy)),int(radious),(0,0,0),-1)
                    if print_flag:
                        # cv2.imwrite('sunglasses.jpg',img)
                        is_sunglass = True

                elif self.hat>0.0 and (temprand - self.sunglass_ratio) <= self.hat_ratio:
                    dirpath = os.path.join(workpath, 'mask_templates/hat/')
                    hatpaths = os.listdir(dirpath)
                    hatpath = dirpath + random.sample(hatpaths,1)[0]
                    hat = Image.open(hatpath)
                    t_width = hat.width
                    t_height = hat.height
                    totalx = 0.0
                    totaly = 0.0
                    count = 1
                    r,g,b,alpha = hat.split()
                    for y in range(t_height):
                        for x in range(t_width):
                            pixel = alpha.getpixel((x,y))
                            if pixel > 0:
                                totalx = totalx + x
                                totaly = totaly + y
                                count = count + 1
                    avrx = int(totalx/count)
                    avry = int(totaly/count)
                    gap = 89 - avrx
                    xstart = int(0 + gap)
                    ystart = int(np.random.rand()*20)
                    xend = xstart + 178
                    yend = ystart + 218
                    tmpimg = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    tmpimg.paste(hat,(xstart,ystart,xend,yend),mask=alpha)
                    img = cv2.cvtColor(np.array(tmpimg),cv2.COLOR_RGB2BGR)
                    if print_flag:
                        # cv2.imwrite('hat.jpg',img)
                        is_hat = True

                elif self.mask > 0.0 :
                    dirpath = os.path.join(workpath, 'mask_templates/respirator/')
                    maskpaths = os.listdir(dirpath)
                    maskpath = dirpath + random.sample(maskpaths,1)[0]
                    mask = Image.open(maskpath)
                    t_width = mask.width
                    t_height = mask.height
                    totalx = 0.0
                    totaly = 0.0
                    count = 1
                    r,g,b,alpha = mask.split()
                    for y in range(t_height):
                        for x in range(t_width):
                            pixel = alpha.getpixel((x,y))
                            if pixel > 0:
                                totalx = totalx + x
                                totaly = totaly + y
                                count = count + 1
                    avrx = int(totalx/count)
                    avry = int(totaly/count)
                    gap = 89-avrx
                    xstart = int(0 + gap)
                    ystart = int(153-avry)
                    xend = xstart + 178
                    yend = ystart + 218
                    tmpimg = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                    tmpimg.paste(mask,(xstart,ystart,xend,yend),mask=alpha)
                    img = cv2.cvtColor(np.array(tmpimg),cv2.COLOR_RGB2BGR)
                    if print_flag:
                        # cv2.imwrite('mask.jpg',img)
                        is_mask = True

                else:
                    pdb.set_trace()

        ## transform
        scale_diff_h = (np.random.rand()*2-1)*self.scale_aug
        scale_diff_w = (np.random.rand()*2-1)*self.scale_aug
        crop_aug_h = self.crop_size*(1+scale_diff_h)
        crop_aug_w = self.crop_size*(1+scale_diff_w)

        trans_diff_h = (np.random.rand()*2-1)*self.trans_aug
        trans_diff_w = (np.random.rand()*2-1)*self.trans_aug

        h, w, _ = img.shape
        ct_x = w/2*(1+trans_diff_w)
        ct_y = (h/2+self.crop_center_y_offset)*(1+trans_diff_h)

        if ct_x < crop_aug_w/2:
            crop_aug_w = ct_x*2 - 0.5
        if ct_y < crop_aug_h/2:
            crop_aug_h = ct_y*2 - 0.5
        if ct_x + crop_aug_w/2 >= w:
            crop_aug_w = (w-ct_x)*2 - 0.5
        if ct_y + crop_aug_h/2 >= h:
            crop_aug_h = (h-ct_y)*2 - 0.5

        #rect = (ct_x-crop_aug_w/2, ct_y-crop_aug_h/2, ct_x+crop_aug_w/2, ct_y+crop_aug_h/2)
        #img = img.resize((self.final_size, self.final_size), box=rect)
        t = int(np.ceil(ct_y-crop_aug_h/2))
        #d = int(np.ceil(ct_y+crop_aug_h/2))
        l = int(np.ceil(ct_x-crop_aug_w/2))
        #r = int(np.ceil(ct_x+crop_aug_w/2))
        img = img[t:int(t+crop_aug_h),l:int(l+crop_aug_w),:]
        img = cv2.resize(img, (self.final_size, self.final_size))

        if print_flag:
            if is_sunglass:
                cv2.imwrite('sunglass.jpg',img)
            elif is_hat:
                cv2.imwrite('hat.jpg',img)
            elif is_mask:
                cv2.imwrite('mask.jpg',img)

        #print(self.flip)
        if np.random.rand() <= self.flip:
            #print('do flip')
            img = cv2.flip(img, 1)

        return img


class ReidAugmentation(object):
    def __init__(self, height, width, re):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.train_transformer = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            normalizer,
        ])

        if re:
            self.train_transformer = T.Compose([
                T.Resize((height, width)),
                T.RandomHorizontalFlip(),
                T.Pad(10),
                T.RandomCrop((height, width)),
                RandomSizedEarser(),
                T.ToTensor(),
                normalizer,
            ])

    def __call__(self, img):
        ## transform
        return self.train_transformer(img)



class ReidAugmentationCV2(object):
    def __init__(self, height, width, re,bri,contrast,brightness_delta=16,contrast_range=(0.8, 1.2), vit=False):
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # self.normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                          std=[0.5, 0.5, 0.5])
        self.height = height
        self.width = width
        self.padding_size = 10
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.re = re
        self.bri = bri
        self.vit = vit
        self.contrast = contrast

    def pad_images(self, img):
        h, w = img.shape[:2]
        width_max = w + 2 * self.padding_size
        height_max = h + 2 * self.padding_size

        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        return img_padded

    def random_crop(self, img):
        h, w = img.shape[:2]
        max_x = w - self.width
        max_y = h - self.height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = img[y:y+self.height,x:x+self.width]
        return crop

    def random_brightness(self, img, binary_mask=None):
        if np.random.rand() <= 0.5:
            img=img.astype(np.float32)
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            if binary_mask is not None:
                delta = delta * binary_mask
            img += delta
        return img

    def random_contrast(self, img, binary_mask=None):
        if np.random.rand() <= 0.5:
            img=img.astype(np.float32)
            alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
            if binary_mask is not None:
                alpha = alpha * binary_mask
            img *= alpha
        return img

    def random_eraser(self, img, sl=0.02, sh=0.2, asratio=0.3, p=0.8, binary_mask=None):
        p1 = random.uniform(-1, 1.0)
        if p1 > p:
            return img
        else:
            H, W = img.shape[:2]
            top, bottom = 0, H
            left, right = 0, W
            if binary_mask is not None:
                hf, wf = np.where(binary_mask[:, :, 0])  # 非0部分
                if hf.shape[0] == 0 or wf.shape[0] == 0:
                    return img
                top = min(hf)
                bottom = max(hf)
                left = min(wf)
                right = max(wf)
            area = (bottom - top) * (right - left)
            gen = True
            times = 0
            while gen:
                times += 1
                if times > 20:
                    return img
                Se = random.uniform(sl, sh)*area
                re = random.uniform(asratio, 1/asratio)
                He = np.sqrt(Se*re)
                We = np.sqrt(Se/re)
                xe = random.uniform(left, right-We)
                ye = random.uniform(top, bottom-He)
                if xe+We <= W and ye+He <= bottom and xe > left and ye > top:
                    x1 = int(np.ceil(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1+We))
                    y2 = int(np.floor(y1+He))
                    part1 = img[y1:y2, x1:x2]
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = np.asarray(part1).astype(np.int8)
                    I[:, :, 0] = Rc
                    I[:, :, 1] = Gc
                    I[:, :, 2] = Bc
                    img[y1:y2, x1:x2] = I
                    return img

    def __call__(self, img, binary_mask=None):
        ## transform
        img = img[:,:,::-1]
        img = cv2.resize(img, (self.width, self.height))
        if np.random.rand() <= 0.5: img = cv2.flip(img, 1)
        if self.bri:
            img=self.random_brightness(img, binary_mask=binary_mask)
        if self.contrast:
            img=self.random_contrast(img, binary_mask=binary_mask)
        img=img.astype(np.uint8)
        if self.re:
            img = self.random_eraser(img, binary_mask=binary_mask)
        img = self.pad_images(img)
        img = self.random_crop(img)

        img = img.transpose((2, 0, 1))
        if not self.vit:
            img = torch.Tensor(img / 255.)
            img = self.normalizer(img)
        else:
            img = torch.Tensor(img)

        return img


class ReidTestAugmentationCV2(object):
    def __init__(self, height, width, vit=False):
        self.height = height
        self.width = width
        self.vit = vit
        self.normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __call__(self, img):
        ## transform
        img = img[:,:,::-1]
        img = cv2.resize(img, (self.width, self.height))
        img = img.transpose((2,0,1))
        if self.vit:
            img = torch.Tensor(img/255.)
            img = self.normalizer(img)
        else:
            img = torch.Tensor(img)
        return img


## Author: liuyakun1
## Time: 20200909
class NoneAugmentation(object):
    def __init__(self, height, width):
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.train_transformer = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            normalizer,
        ])

    def __call__(self, img):
        ## transform
        return self.train_transformer(img)


class ReidTestAugmentation(object):
    def __init__(self, height, width):

        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.test_transformer = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            normalizer,
        ])

    def __call__(self, img):
        ## transform
        return self.test_transformer(img)


class RandomSizedEarser(object):
    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.8):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1, 1.0)
        W = img.size[0]
        H = img.size[1]
        area = H * W

        if p1 > self.p:
            return img
        else:
            gen = True
            while gen:
                Se = random.uniform(self.sl, self.sh)*area
                re = random.uniform(self.asratio, 1/self.asratio)
                He = np.sqrt(Se*re)
                We = np.sqrt(Se/re)
                xe = random.uniform(0, W-We)
                ye = random.uniform(0, H-He)
                if xe+We <= W and ye+He <= H and xe>0 and ye>0:
                    x1 = int(np.ceil(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1+We))
                    y2 = int(np.floor(y1+He))
                    part1 = img.crop((x1, y1, x2, y2))
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = Image.new('RGB', part1.size, (Rc, Gc, Bc))
                    img.paste(I, (x1, y1))
                    return img

class RandomSizedEarserCV2(object):
    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.8):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, img):
        p1 = random.uniform(-1, 1.0)
        H, W = img.shape[:2]
        area = H * W

        if p1 > self.p:
            return img
        else:
            gen = True
            while gen:
                Se = random.uniform(self.sl, self.sh)*area
                re = random.uniform(self.asratio, 1/self.asratio)
                He = np.sqrt(Se*re)
                We = np.sqrt(Se/re)
                xe = random.uniform(0, W-We)
                ye = random.uniform(0, H-He)
                if xe+We <= W and ye+He <= H and xe>0 and ye>0:
                    x1 = int(np.ceil(xe))
                    y1 = int(np.ceil(ye))
                    x2 = int(np.floor(x1+We))
                    y2 = int(np.floor(y1+He))
                    part1 = img[y1:y2, x1:x2]
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    I = np.asarray(part1).astype(np.int8)
                    I[:,:,0] = Rc
                    I[:,:,1] = Gc
                    I[:,:,2] = Bc
                    img[y1:y2, x1:x2] = I
                    return img

class BodySplit(object):
    def __init__(self,extra_info=None,bg_type=0,aug_type=-1,split_prob=0.5):
        """
        aug: for half augmentation
            -1: no aug
            0:only return aug img wo padding
            1:add padding
        """
        self.extra_info=extra_info
        self.bg_type=bg_type
        self.mean=[104,116,124]
        self.aug_type=aug_type
        self.debug=1
        self.split_prob=split_prob
        if extra_info is not None:
            print('read extra_info')
            if not os.path.exists(extra_info):
                extra_info = extra_info.replace('/mnt/lustre/share', '/mnt/lustre/share_data')  # sh1986
            f = open(extra_info, 'r')
            self.info = dict()
            for line in f.readlines():
                items = line.strip('\n').split(' ')
                key = items[0]#.split('/')[-1]
                #print(key)
                #import pdb;pdb.set_trace()
                self.info[key] = (float(items[-4]), float(items[-3]), float(items[-2]), float(items[-1]))
            f.close()
        print('Done')

    def blind_split(self, raw_img):
        w, h = raw_img.size
        if np.random.rand() <= self.split_prob:
          if h > w * 0.7:
            mid = max(w*0.7, int(round(h*0.35)))
            cut = np.random.randint(low=mid, high=h)
            # raw_img = raw_img[:cut]
            raw_img = raw_img.crop((0, 0, w, cut))
        return raw_img

    def __call__(self, raw_img,fname):
        if self.extra_info is None:
            return raw_img
        #name = fname.split('/')[-1]

        if fname not in self.info.keys():
            print("="*80)
            print(fname)
            return raw_img

        head, _, mid, foot = self.info[fname]
        if head<0:
            return raw_img
        if self.aug_type==0: #crop from ori img
            if np.random.rand() <= self.split_prob and mid<=1 and mid>0.15:
                h, w, c = raw_img.shape
                mid=int(mid*h)
                raw_img=raw_img[:mid]
                if self.debug:
                    cv2.imwrite('test_split.jpg',raw_img)
                    print('save debug img to test_split.jpg')
                    self.debug=0
            return raw_img
        if self.aug_type==2: #random
            if np.random.rand() <= self.split_prob and mid<=1 and mid>0.15:
                h, w, c = raw_img.shape
                mid=max(int(mid*h)//2,1)
                crop_line=np.random.randint(low=mid,high=h)
                raw_img=raw_img[:crop_line]
            return raw_img
        if self.aug_type==3: #blind_split
            return self.blind_split(raw_img)

        h, w, c = raw_img.shape

        img = np.zeros((3*h, w, c))
        if self.bg_type == 0:
            #add mean -->background
            img[:,:,0],img[:,:,1],img[:,:,2]=np.mean(raw_img[:,:,0]),np.mean(raw_img[:,:,1]),np.mean(raw_img[:,:,2])
        if self.bg_type == 1: #black
            pass
        if self.bg_type == 2:#imagenet mean
            img[:,:,0],img[:,:,1],img[:,:,2]=self.mean[0],self.mean[1],self.mean[2]
        img[h*3 // 2:h + (h*3 // 2), :, :] = raw_img
        start = int((head + 1.5) * h)
        end = int((foot + 1.5) * h)

        if start < end:
            img=img[start:end].astype(np.uint8)

            if self.aug_type==1: #for data augmentation
                if np.random.rand() <= self.split_prob and (mid-head)>0.15:
                    mid=int((mid-head)*h)
                    img=img[:mid]
        return img


class BodyAlign(object):
    def __init__(self, height, width, split_list_file, crop_prob=0, prefix='', aug_type=0, correction=False):
        """
        split_info: path head, neck, waist, foot
        bg_type: 0 stands for img_mean, 1 stands for black, 2 stands for mean of imagenet
        """
        self.prefix = prefix
        self.split_list_file = split_list_file

        # self.mean = [0.15714108, 0.27813716, 0.56472176]
        # self.var = [0.00058989, 0.0009067, 0.00128566]

        self.aug_type = aug_type
        self.crop_prob = crop_prob
        self.imagenet_mean = [104, 116, 124]
        self.height = height
        self.width = width
        # template reference: aolai/indoor/aolai+20190218_10.254.250.96+video_2019_02_16_14_29.mov+221488+8+800_0_0_133_331_0.93.png 0.000000 0.156250 0.468750 0.968750
        self.template_shape = (width, height)

        self.template_split_info = [0, 0.15625, 0.46875, 0.96875]  # head, neck, waist, foot
        self.template_point = np.array([
            [0, self.template_split_info[0]], [width, self.template_split_info[0]],
            [0, self.template_split_info[1] * height], [width, self.template_split_info[1] * height],
            [0, self.template_split_info[2] * height], [width, self.template_split_info[2] * height],
            [0, self.template_split_info[3] * height], [width, self.template_split_info[3] * height]
        ], dtype=np.float32)

        self._parse_split_info(correction=correction)

        print('[body align]correction:{}, height:{}, width:{}, aug_type {}, crop_prob:{}'.
              format(correction, self.height, self.width, self.aug_type, self.crop_prob))

    def _is_split_info_valid(self, head, neck, waist, foot):
        if not head < 0.3:
            return False
        if not (head < waist or neck < waist or waist < foot):
            return False
        return True

    def _correct_split_info(self, head, neck, waist, foot, version=0):
        standard_head = self.template_split_info[0]
        standard_neck = self.template_split_info[1]
        standard_waist = self.template_split_info[2]
        standard_foot = self.template_split_info[3]
        margin_head_neck = self.template_split_info[1] - self.template_split_info[0]
        margin_neck_waist = self.template_split_info[2] - self.template_split_info[1]
        margin_waist_foot = self.template_split_info[3] - self.template_split_info[2]

        delta = 0
        if not head < 0.3:
            delta = head - standard_head
            head = standard_head

        if version == 0:
            if head < waist or neck < waist or waist < foot:
                neck = max(head + margin_head_neck, neck - delta)
                waist = max(neck + margin_neck_waist, waist - delta)
                foot = max(waist + margin_waist_foot, foot - delta)
                return head, neck, waist, foot
        elif version == 1:
            # if head < waist or neck < waist or waist < foot:
            #     if neck - delta <= head:
            #         neck = head + margin_head_neck
            #     else:
            #         neck -= delta
            #     if waist - delta <= neck:
            #         waist = neck + margin_neck_waist
            #     else:
            #         waist -= delta
            #     if foot - delta <= waist:
            #         foot = waist + margin_waist_foot
            #     else:
            #         foot -= delta
            #     return head, neck, waist, foot
            raise ValueError('may be wrong')
        else:
            raise ValueError('unknown correction version:{}'.format(version))

        if not head < neck:
            neck = max(head + margin_head_neck, standard_neck)
        if not neck < waist:
            waist = max(neck + margin_neck_waist, standard_waist)
        if not waist < foot:
            foot = max(waist + margin_waist_foot, standard_foot)
        return head, neck, waist, foot

    def _parse_split_info(self, correction=False):
        print('parsing split_list_file:{}'.format(self.split_list_file))
        if not os.path.exists(self.split_list_file):
            self.split_list_file = self.split_list_file.replace('/mnt/lustre/share', '/mnt/lustre/share_data')  # sh1986
        assert os.path.exists(self.split_list_file), 'file not exist!! [{}]'.format(self.split_list_file)
        f = open(self.split_list_file, 'r')
        self.info = dict()
        correc_cnt = 0
        for line in f.readlines():
            items = line.strip('\n').split(' ')
            path = items[0]
            key = os.path.join(self.prefix, path)
            # head, neck, waist, foot
            head, neck, waist, foot = float(items[-4]), float(items[-3]), float(items[-2]), float(items[-1])
            if correction:
                if not self._is_split_info_valid(head, neck, waist, foot):
                    correc_cnt += 1
                    ori_head, ori_neck, ori_waist, ori_foot = head, neck, waist, foot
                    head, neck, waist, foot = self._correct_split_info(ori_head, ori_neck, ori_waist, ori_foot)
                    # print('{}: need correction for split_info\n[{}, {}, {}, {}] => [{}, {}, {}, {}]'
                    #       .format(path, ori_head, ori_neck, ori_waist, ori_foot, head, neck, waist, foot))
            self.info[key] = (head, neck, waist, foot)
        if correc_cnt > 0:
            print("{} split_info items were made correction".format(correc_cnt))
        f.close()

    @staticmethod
    def affine_image(img, src_points, dst_points, shape,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_LINEAR):
        """

        :param img:
        :param src_points:
        :param dst_points:
        :param shape:
        :param borderMode:  cv2.BORDER_CONSTANT
        :param borderValue: 0  # 填充黑边
        :param flags:
        :return:
        """
        trans_matrix = kestrel_get_similar_matrix(src_points, dst_points)
        trans_matrix = np.concatenate((trans_matrix, [[0, 0, 1]]), axis=0)
        return cv2.warpPerspective(img, trans_matrix, shape,
                                   borderMode=borderMode, borderValue=borderValue, flags=flags), trans_matrix

    def _get_crop_region(self, img, aug_type):
        head, neck, waist, foot = self.template_split_info
        crop_top, crop_bottom = -1, -1
        if aug_type == 0:  # crop from ori img
            h, w, c = img.shape
            waist_height = int(waist * self.height)
            crop_top = np.random.randint(waist_height, waist_height + int(0.2 * self.height))
            crop_bottom = h
        elif aug_type == 2:  # random
            h, w, c = img.shape
            crop_height = max(int(waist * h) // 2, 1)
            crop_top = np.random.randint(low=crop_height, high=h)
            crop_bottom = h
        elif aug_type == 3:  # random crop_aug for 'top' or 'bottom'
            crop_style = np.random.choice(['top', 'bottom'], 1)
            h, w, c = img.shape
            if crop_style == 'top':
                crop_height = max(int(neck * h), 1)
                crop_top = 0
                crop_bottom = np.random.randint(low=0, high=crop_height)
            else:
                return self._get_crop_region(img, aug_type=2)
        return [crop_top, crop_bottom]

    def __call__(self, raw_img, fname, specific_info=None, return_mask=False):
        h, w, c = raw_img.shape
        output = {}

        if specific_info is not None:
            # debug-mode
            head, neck, waist, foot = specific_info
        else:
            if fname not in self.info.keys():
                print("split_info not found".center(20, '='))
                print(fname)
                return output
            head, neck, waist, foot = self.info[fname]

        src_point = np.array([
            [0, head * h], [w, head * h],
            [0, neck * h], [w, neck * h],
            [0, waist * h], [w, waist * h],
            [0, foot * h], [w, foot * h],
        ], dtype=np.float32)

        affined_img, trans_matrix = self.affine_image(raw_img, src_point, self.template_point,
                                                      shape=self.template_shape)

        if np.random.rand() <= self.crop_prob:
            crop_region = self._get_crop_region(affined_img, aug_type=self.aug_type)
        else:
            crop_region = [-1, -1]
        if 0 < crop_region[0] < crop_region[1]:
            affined_img[crop_region[0]: crop_region[1], ...] = 0

        output['affined_image'] = affined_img
        # 'theta': theta_tensor

        if return_mask:
            # do align&crop as raw_img
            fake_img = np.full(raw_img.shape, 125).astype(np.uint8)
            affined_fake_img, _ = self.affine_image(fake_img, src_point, self.template_point,
                                                    shape=self.template_shape)
            if len(affined_fake_img.shape) == 3:
                lum = np.max(affined_fake_img, axis=2)
            else:
                lum = affined_fake_img
            binary_mask = np.ones_like(lum)
            binary_mask = np.where(lum == 0, 0, binary_mask)
            if 0 < crop_region[0] < crop_region[1]:
                binary_mask[crop_region[0]: crop_region[1]:, ...] = 0
            output['binary_mask'] = binary_mask

        return output
