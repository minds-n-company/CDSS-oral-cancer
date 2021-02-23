import logging
import numpy as np
import pandas as pd
import os, glob, time, datetime, re
import pickle
import gzip
import copy
from pytz import timezone, utc
# import math
import random
import torch
# import torchvision
# from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from multiprocessing import Pool

import torch.nn as nn
# from torchvision.transforms import transforms
# from theconf import Config as C
# from utils import get_logger # TODO 
# import albumentations as A
import torchvision.transforms as A
from PIL import Image
# from torchsummary import summary

import matplotlib.pyplot as plt
# from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.tensorboard import SummaryWriter


disease = ['Sinusitis','Oral_cancer'][1]
rm_lowq = False
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

def get_logger(name, level=logging.DEBUG, resetlogfile=False, path='log'):
    fname = os.path.join(path, name+'.log')
    os.makedirs(path, exist_ok=True) 
    if resetlogfile :
        if os.path.exists(fname):
            os.remove(fname) 
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(fname)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

KST = timezone('Asia/Seoul')
today_datetime = datetime.datetime.utcnow().strftime("%y%m%d_%Hh%Mm")    
logger = get_logger(f'{disease}_EfficientNet_{today_datetime}', resetlogfile=True)
logger.setLevel(logging.INFO)

def prepare_metatable(filenames):
    cvdf = pd.DataFrame(filenames)
    cvdf.columns = ['filename']
    cvdf['patientID'] = cvdf['filename'].apply(lambda x : x.split('/')[-1].split('_')[0])

    if disease=='Oral_cancer':
        splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').replace('.JPG','').split('_')
        cvdf['label_org'] = cvdf['filename'].apply(lambda x : splitUnderbar(x)[-1])
        # print(cvdf['label_org'].value_counts())
        # assert sum(~cvdf['label_org'].isin(['N','B','C']))==0, 'Found Unknown Diagnosis' # TODO
        cvdf['label'] = 0
        cvdf.at[cvdf['label_org']=='C','label'] = 1

    cvdf['FOLD'] = np.nan

    oldcolumns = cvdf.columns.tolist()
    cvdf['index'] = cvdf.index
    cvdf = cvdf[['index']+oldcolumns]
    # print(cvdf.head())
    return cvdf

def save_validationlist(root='.'):
    if disease=='Oral_cancer':
        data_dir = "3_검증/원천데이터"
        # list up filenames of valid data
        totalfiles = glob.glob(os.path.join(root,data_dir,"*.jpg"))
        totalfiles.extend(glob.glob(os.path.join(root,"**","*.JPG")))
        invalidfiles = glob.glob(os.path.join(root,data_dir,"*.bmp"))
        filenames = totalfiles.copy()

        if len(invalidfiles)>1:
            logger.warn(invalidfiles)
            filenames.remove(invalidfiles)
        # splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').replace('.JPG','').split('_')
        
        totalpatients = set([re.sub(r'^.+/','',x).split('_')[0] for x in totalfiles])
        patients = set([re.sub(r'^.+/','',x).split('_')[0] for x in filenames])

    logger.info('='*10+' '*20 +'START ANALYSIS'+' '*20+ '='*10)
    logger.info(f'No. of total datasets : {len(totalfiles)} files, {len(totalpatients)} patients') # expected :  12400 files
    logger.info(f'No. of valid datasets : {len(filenames)} files, {len(patients)} patients (excluded wrong format)') 
    
    cvdf = prepare_metatable(filenames if rm_lowq else totalfiles)

    n_folds = 10

    if disease=='Oral_cancer':

        plen = len(filenames) if disease == 'Sinusitis' else len(patients if rm_lowq else totalpatients)

        pat_df = cvdf.loc[~cvdf['patientID'].duplicated(),:].reset_index()

        logger.info(f'----- Split patients for {n_folds} Cross-validation')
        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        cvdf['FOLD'] = np.nan

        for ii, (train_pidx, test_pidx) in enumerate(skf.split(pat_df['patientID'],pat_df['label'])):
            train_pat = pat_df['patientID'].iloc[train_pidx]
            test_pat = pat_df['patientID'].iloc[test_pidx]
            # record fold index
            cvdf.loc[cvdf['patientID'].isin(test_pat),cvdf.columns=='FOLD']= ii
            cvdf[f'FOLD{ii}_testset'] = 0
            cvdf.loc[cvdf['patientID'].isin(test_pat),cvdf.columns==f'FOLD{ii}_testset'] = 1

    # save metadata
    filelist_dir = os.path.join(root,'inputlist')
    os.makedirs(filelist_dir, exist_ok=True) 

    cvdf.to_csv(os.path.join(filelist_dir,"input_crossvalidation_table.csv"),index=False)
    cvdf[['index','filename']].to_csv(os.path.join(filelist_dir,"input_filenames_total.csv"),index=False)
    for i in range(n_folds):
        cvdf.loc[cvdf[f'FOLD{i}_testset']==1,'filename'].to_csv(os.path.join(filelist_dir,f"input_filenames_fold{i}.csv"),index=False)

    # statistics
    logger.info(f'----- Data statistics') 
    
    if disease=='Oral_cancer':
        logger.info('-- Original label x label')
        logger.info(pd.crosstab(cvdf['label_org'], cvdf['label'], margins=True))

        logger.info('-- Patient frequency by FOLD')
        logger.info(cvdf.groupby(cvdf['FOLD']).patientID.nunique())
        logger.info('-- Image frequency by FOLD')
        logger.info(cvdf['FOLD'].value_counts())
        labelfreq = pd.crosstab(cvdf['FOLD'],cvdf['label'], margins=True)
        labelfreq_ratio = pd.crosstab(cvdf['FOLD'],cvdf['label'], margins=True, normalize='index')
        labelfreq.to_csv(os.path.join(filelist_dir,f"label_freq_byfold.csv"))
        labelfreq_ratio.to_csv(os.path.join(filelist_dir,f"label_freq_ratio_byfold.csv"), float_format = '%.2f')
        logger.info(f'-- Label frequency by fold') 
        logger.info(labelfreq)
        logger.info(f'-- Label frequency (ratio) by fold') 
        logger.info(labelfreq_ratio)    
        
import multiprocessing

class ImageDataset(Dataset):
    def __init__(self, root='.', input_csv='inputlist/input_filenames_fold', fold_num=0, data_type='train', carrydata=True, transform=None, cropped=True):
        super(ImageDataset, self).__init__()
        self.root = root
        self.input_csv = input_csv
        self.fold_num = fold_num
        self.data_type = data_type # train, val, test
        self.carrydata = carrydata
        self.transform = transform
        self.cropped = cropped
        logger.info('--'*20)
        logger.info(f"- Build {self.data_type} dataset") 
        logger.info('-- Transform')
        logger.info(self.transform)
        logger.info('-- ')

        n_folds = 10 

        train_fold = list(range(n_folds))
        val_fold = train_fold[fold_num-1]
        train_fold.remove(fold_num) # test set
        train_fold.remove(val_fold) # validation set

        if data_type=="train":
            self.filenames = []
            for i in train_fold:
                fl_i = pd.read_csv(f'{input_csv}{i}.csv')['filename'].tolist()
                self.filenames.extend(fl_i)
        elif data_type=="val":    
            self.filenames = pd.read_csv(f'{input_csv}{val_fold}.csv')['filename'].tolist()
        elif data_type=="test":    
            self.filenames = pd.read_csv(f'{input_csv}{fold_num}.csv')['filename'].tolist()

        logger.info(f" Read {len(self.filenames)} files : '{self.filenames[0]}', etc.") # tmp

        self.imagedata = []
        self.targetdata = [] # binary class (0 to 1)
        self.orginal_targetdata = [] # multiclass (0 to 3 or N, C, B)

        save_dir = 'preprocessed_data'
        os.makedirs(save_dir, exist_ok=True)

        if self.carrydata: 
            # means , stds = [], []

            n_cpu = multiprocessing.cpu_count()
            used_th = 20
            logger.info(f'no. cpu : {n_cpu}, use {used_th} threads')
            pool = multiprocessing.Pool(processes=used_th)
            result = pool.map(self.read_data, self.filenames)
            pool.close()
            pool.join()

            for i in result :
                if disease=='Sinusitis':
                    self.imagedata.extend(i[1]) 
                elif disease=='Oral_cancer':
                    self.imagedata.append(i[1]) 
                    self.orginal_targetdata.append(i[2])
            unexpected_target = [ i for i in self.orginal_targetdata if i not in ['N','C','B']]
            if len(unexpected_target)>1 :
                logger.warn('Unexpected target :')
                logger.warn(unexpected_target)
            assert len(unexpected_target)==0, unexpected_target 
            # mn , std = np.mean(imgs), np.std(imgs)
            # means.append(mn)
            # stds.append(std)
            # dichotomize target
            if disease=="Oral_cancer":
                self.targetdata = [1 if x=='C' else 0 for x in self.orginal_targetdata]

            logger.info(f' No. of datasets loaded  : {len(self.imagedata)} images')
            logger.info(f' No. of datasets loaded  : {len(self.targetdata)} target')
            from collections import Counter
            logger.info(Counter(self.targetdata))
            logger.info(Counter(self.orginal_targetdata))
            # if data_type=='train': # TODO 
            #     self.avg_mean = sum(means) / len(means) 
            #     self.avg_std = sum(stds) / len(stds) 
            #     logger.info(f' Averaged intensity mean {self.avg_mean},  Averaged intensity standard deviation {self.avg_std} ')

    def __len__(self):
        return len(self.imagedata)

    def __getitem__(self, index):
        if self.carrydata:
            img = self.imagedata[index]
            target = self.targetdata[index]
        else:
            _, img, target = self.read_data(self.filenames[index])
        if self.transform:
            # ref: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            img = Image.fromarray(img, 1 if disease=='Sinusitis' else 'RGB') 
            img = self.transform(img)
        return img, target

    def patients_length(self):
        return len(self.filenames)

    def read_data(self, fn, pkl_dir='preprocessed_data'):

        if disease =="Oral_cancer":
            splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').replace('.JPG','').split('_')
            targets = splitUnderbar(fn)[-1]

        patient_id = fn.split('/')[-1].split('_')[0]
        
        pkl_fname = os.path.join(pkl_dir, f"{fn.replace('.dcm', '').replace('.jpg','').replace('.JPG','').replace('./', '').replace('/', '__')}.pkl")
        if os.path.exists(pkl_fname):
            # load pickle file
            logger.info(f' > Load preprocessed data : {pkl_fname}')
            try:
                with gzip.open(pkl_fname,'rb') as f:
                    [patient_id, imgs, targets] = pickle.load(f)
                    # [patient_id, imgs, targets] = [np.copy(v) for v in pickle.loads(f)] #TODO
                    # img_dump = {k: np.copy(v) for k, v in msgpack.loads(dump, raw=False).items()} #TODO
                    # https://discuss.pytorch.org/t/userwarning-the-given-numpy-array-is-not-writeable/78748/8
            except :
                logger.warn(f'remove preprocessed data & re-process : {pkl_fname}')
                os.remove(pkl_fname)
                patient_id, imgs, targets = self.read_data(fn)

        else: 
            logger.info(f' > Load & preprocess data : patient # {patient_id}')
            img_format = fn.split('.')[-1]
            if img_format == 'dcm' or img_format == 'DCM' :        
                d = dicom.read_file(fn)
                org_img = d.pixel_array
            elif img_format == 'jpg' or img_format == 'jpeg' or img_format == 'JPG' or \
            img_format == 'png' or img_format == 'PNG':
                if disease =="Oral_cancer":
                    imgs = np.array(Image.open(fn)) #RGB
                    # imgs = org_img/255

            logger.info(f' >> Input image shape : {imgs.shape[0]} x {imgs.shape[1]} x {imgs.shape[2]}') # sinus : 2 x 318 x 318, oral: 480 x 640 x 3

            # elif disease=='Oral_cancer':
            #     plt.imsave('./png_input/'+patient_id +'-original.png', org_img)

            # save pickle file
            objs = [patient_id, imgs, targets]
            with gzip.open(pkl_fname, 'wb') as f:
                pickle.dump(objs, f)

        return patient_id, imgs, targets

    def scale_minmax(self, img):
        img_minmax = img.copy()
        
        if np.ndim(img) == 3:
            for d in range(3):
                img_minmax[d] = self.scale_minmax(img[d])
        else:
            img_minmax = ((img - np.min(img)) / (np.max(img) - np.min(img))).copy()
            
        return img_minmax
    

def get_dataloaders(args, dataset_class, batch, root, fold_num=0, multinode=False, augment=False):

    logger.info(f'----- Get data loader : Vailidation No. {fold_num}')

    # augmentation
    transform_test = A.Compose(
        [   
            A.Resize((300, 300)), #  not A.Resize(224)
            A.ToTensor(),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if disease=='Oral_cancer' else None,
            # ToTensorV2()
        ]
    )
    
    if augment:
        transform_train = A.Compose(
            [
                # A.ToFloat(max_value=None),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                # A.RandomCrop(height=128, width=128),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(300,300),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #for RGB # TODO : normalize
                ToTensorV2()
            ]
        )
    else: 
        transform_train = transform_test

    # prepare training dataset using transform_train
    if args.trained_model is not None:
        train_dataset = None
    else:    
        train_dataset = dataset_class(data_type='train', fold_num=fold_num, transform=transform_train)

    # prepare validation(for hyperparam. tuning), test sets using transform_test
    if args.trained_model is not None:
        val_dataset = None
    else:    
        val_dataset = dataset_class(data_type='val', fold_num=fold_num, transform=transform_test)
    test_dataset = dataset_class(data_type='test', fold_num=fold_num, transform=transform_test)

    train_sampler, valid_sampler, test_sampler = None, None, None 
    trainloader, validloader, testloader = None, None, None 

    # define sampler for imbalanced data
    if args.trained_model is None :
        target_list = torch.tensor(train_dataset.targetdata)
        target_list = target_list[torch.randperm(len(target_list))]
        class_sample_count = np.array([len(np.where(target_list==t)[0]) for t in np.unique(target_list)])
        class_weights = 1./class_sample_count 
        class_weights = torch.from_numpy(class_weights).type('torch.DoubleTensor')

        logger.info('class_weights : ')
        logger.info(class_weights)

        class_weights_all = class_weights[target_list]

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        # data loader
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=True if train_sampler==None else False, num_workers=2, pin_memory=False,
            sampler=train_sampler, drop_last=True)

        validloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch, shuffle=False, pin_memory=True,
            sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch, shuffle=False, pin_memory=True,
        sampler=test_sampler, drop_last=False)

    return test_dataset, trainloader, validloader, testloader
