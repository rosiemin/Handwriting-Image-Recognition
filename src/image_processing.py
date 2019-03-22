import math, glob, sys, os, re
from PIL import Image, ImageOps
from src.read_data import IAMLoadData
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pdb

class IAM_imageprocess():
    word = 'word'
    line = 'line'
    color = 255

    def __init__(self, df, cutoff_height = 100, cutoff_width = 250, desired_height = 32, desired_width = 64, size = 1000, rootDir = 'data/words'):
        self.df = df
        self.rootDir = rootDir
        self.cutoff_height = cutoff_height
        self.cutoff_width = cutoff_width
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.size = size

    def IAM_images(self):
        # pdb.set_trace()
        if IAM_imageprocess.word in self.rootDir:
            if not os.path.isfile('data/words_image_size_df.csv'):
                self.max_h_w_image()
            self.subset_by_size('data/words_image_size_df.csv')
        elif IAM_imageprocess.line in self.rootDir:
            if not os.path.isfile('data/lines_image_size_df.csv'):
                self.max_h_w_image()
            self.subset_by_size('data/lines_image_size_df.csv')
        self.remove_nonalpha()
        self.resize_images()
        X_train, X_test, y_train, y_test = self.train_test_split()

        return X_train, X_test, y_train, y_test, self.df

    def max_h_w_image(self):
        print("Parsing out Image heights and weights...")
        print("Retrieving file names...")
        self.filelist = []

        for i in tqdm(self.df['file_path']):
            self.filelist.append(self.rootDir+"/"+i)

        max_pixel_h = []
        max_pixel_w = []

        for image in self.filelist:
            old_im = Image.open(image)
            max_pixel_h.append(old_im.size[1])
            max_pixel_w.append(old_im.size[0])

        image_size_df = pd.DataFrame(columns = ['widths', 'heights'])

        image_size_df['widths'] = max_pixel_w
        image_size_df['heights'] = max_pixel_h

        self.df['height'] = image_size_df['heights']
        self.df['width'] = image_size_df['widths']

        if IAM_imageprocess.word in self.rootDir:
            image_size_df.to_csv('data/words_image_size_df.csv')
        elif IAM_imageprocess.line in self.rootDir:
            image_size_df.to_csv('data/lines_image_size_df.csv')


    def subset_by_size(self, size_df_path = None):

        size_df = pd.read_csv(size_df_path)
        self.df['height'] = size_df['heights']
        self.df['width'] = size_df['widths']

        self.full_df = self.df.copy()

        self.df = self.df[(self.df['height'] <= self.cutoff_height) & (self.df['width'] <= self.cutoff_width)].reset_index(drop=True)
        self.df = self.df[(self.df['height']>10) & (self.df['width']>10)].reset_index(drop = True)
        print(f"Reduced dataframe from {len(self.full_df)} to {len(self.df)}")

        return self.full_df, self.df

    def remove_nonalpha(self):
        a = len(self.df)
        idx_lst = []
        for idx, word in enumerate(self.df['target']):
            if word.isalpha():
                pass
            else:
                idx_lst.append(idx)

        self.df = self.df.drop(idx_lst, axis = 0).reset_index(drop=True)
        print(f"Reduced dataframe from {a} to {len(self.df)}")

        return self.df

    def resize_images(self):
        print(f"Resizing {len(self.df)} Images...")

        self.filelist = []

        for i in tqdm(self.df['file_path']):
            self.filelist.append(self.rootDir+"/"+i)
        # parses through your file list and processes each image
        self.vect_img_lst = []
        for id, image in zip(self.df['id'],tqdm(self.filelist)):
            newfilepath = "data/pad_img/words/"+id+ f"-RESIZE-{self.desired_width}x{self.desired_height}.png"
            if os.path.isfile(newfilepath):
                new_im = ImageOps.grayscale(Image.open(newfilepath))
                self.vect_img_lst.append(np.array(new_im))
            else:
                old_im = ImageOps.grayscale(Image.open(image))
                img_width = old_im.size[0]
                img_height = old_im.size[1]
                if img_width < self.cutoff_width and img_width < self.desired_width:
                    delta_w = self.desired_width - img_width
                    pad_w = (delta_w//2, 0, delta_w-(delta_w//2), 0)
                    old_im = ImageOps.expand(old_im, pad_w, 255)
                if img_height < self.cutoff_height and img_height < self.desired_height:
                    delta_h = self.desired_height - img_height
                    pad_h = (0, delta_h//2, 0, delta_h-(delta_h//2))
                    old_im = ImageOps.expand(old_im, pad_h, color)
                # pad all images whether we added padding or not
                extra_pad = (10, 10, 10, 10)
                old_im = ImageOps.expand(old_im, extra_pad, color)
                #resize images to be our desired size
                new_im = old_im.resize((self.desired_width, self.desired_height))
                self.vect_img_lst.append(np.array(new_im))

                new_im.save(newfilepath)

    def train_test_split(self):
        print(f'Randomly selecting {self.size} samples...')
        print("Performing train test split")
        subsample = random.sample(range(len(self.vect_img_lst)), self.size)
        X = [self.vect_img_lst[i] for i in subsample]
        y = self.df[['id','target']].values[subsample]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    words = IAMLoadData('data/words.txt')
    df = words.load_data()

    clean = IAM_imageprocess(df)

    X_train, X_test, y_train, y_test, df = clean.IAM_images()
