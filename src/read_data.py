import numpy as np
import pandas as pd
import pdb
import math, glob, sys, os, re
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

class IAMLoadData():
    word = 'word'


    def __init__(self, file):
        self.file = file
        self.load_data()

    def load_data(self):
        '''Only need to instantiate the class and use this method
        to get the dataframe loaded in. '''


        print("Loading data from word images...")
        self.get_ids()
        self.get_word_targets()
        self.cleaning_targets()
        self.list_to_df()
        self.remove_nonalpha()
        return self.df

    def get_ids(self):
        self.id_lst = []
        with open(self.file) as f:
            for line in f:
                word = line.split(' ')[-1].strip() #removes any file that has an upper case
                if any(x.isupper() for x in word):
                    pass
                else:
                    self.id_lst.append(line.split(' ')[0])

    def get_word_targets(self):
        self.target_words = []
        with open(self.file) as f:
            for line in f:
                word = line.split(' ')[-1].strip()
                if any(x.isupper() for x in word):
                    pass
                else:
                    self.target_words.append(word)


    def cleaning_targets(self):
        self.target_lst = []
        if len(self.id_lst) == len(self.target_words):
            print("Cleaning target for df:")
            for id, target in zip(self.id_lst, self.target_words):
                path = id.replace("-", "/")[:4] + id[:-6] + f'/{id}.png'
                line_id = id[:-3]
                self.target_lst.append([id, target, path, line_id])
        else:
            print("The ID length and target length do not match")

    def list_to_df(self):
        col_names = ['id','target', 'file_path', 'line_id']
        self.df = pd.DataFrame(self.target_lst, columns = col_names)
        if IAMLoadData.word in self.file:
            error_id_lst = (self.df['id']=='a01-117-05-02') | (self.df['id']=='r06-022-03-05')
            self.df = self.df[~error_id_lst].reset_index(drop=True)


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

if __name__ == '__main__':
    words = IAMLoadData('data/words.txt')
    df = words.df
