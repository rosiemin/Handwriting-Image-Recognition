import numpy as np
import pandas as pd
import pdb

class IAMLoadData():
    word = 'word'
    line = 'line'

    def __init__(self, file):
        self.file = file

    def load_data(self):
        '''Only need to instantiate the class and use this method
        to get the dataframe loaded in. '''

        if IAMLoadData.word in self.file:
            print("Loading data from word images...")
            self.get_ids()
            self.get_word_targets()
            self.cleaning_targets()
            self.list_to_df()
            return self.df
        elif IAMLoadData.line in self.file:
            print("Loading data from line images...")
            self.get_ids()
            self.get_line_targets()
            self.cleaning_targets()
            self.list_to_df()
            return self.df

    def get_ids(self):
        self.id_lst = []
        with open(self.file) as f:
            for line in f:
                self.id_lst.append(line.split(' ')[0])

    def get_word_targets(self):
        self.target_words = []
        with open(self.file) as f:
            for line in f:
                self.target_words.append(line.split(' ')[-1].strip().lower())

    def get_line_targets(self):
        self.target_lines = []
        with open(self.file) as f:
            for line in f:
                word_line = line.split(' ')[-1].lower()
                self.target_lines.append(word_line.replace("|", " ").strip())

    def cleaning_targets(self):
        self.target_lst = []
        if IAMLoadData.word in self.file:
            for id, target in zip(self.id_lst, self.target_words):
                path = id.replace("-", "/")[:4] + id[:-6] + f'/{id}.png'
                line_id = id[:-3]
                self.target_lst.append([id, target, path, line_id])
        elif IAMLoadData.line in self.file:
            for id, target in zip(self.id_lst, self.target_lines):
                path = id.replace("-", "/")[:4] + id[:-3] + f'/{id}.png'
                line_id = id
                self.target_lst.append([id, target, path, line_id])

    def list_to_df(self):
        col_names = ['id','target', 'file_path', 'line_id']
        self.df = pd.DataFrame(self.target_lst, columns = col_names)
        if IAMLoadData.word in self.file:
            error_id_lst = (self.df['id']=='a01-117-05-02') | (self.df['id']=='r06-022-03-05')
            self.df = self.df[~error_id_lst].reset_index(drop=True)

        return self.df


if __name__ == '__main__':
    words = IAMLoadData('data/words.txt')
    words.get_ids()
    words.get_word_targets()
    words.cleaning_targets()
    words.list_to_df()


    lines = IAMLoadData('data/lines.txt')
    lines.get_ids()
    lines.get_line_targets()
    lines.cleaning_targets()
    lines_df = lines.list_to_df()
