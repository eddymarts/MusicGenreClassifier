import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import *
import pandas as pd
import os
import ast

class MusicData:
  """
  Class that implements torch.utils.data.Dataset
  """
  def __init__(self, device=torch.device("cpu")):
    self.device = device
    self.get_data()
    
  def get_data(self, subset='large'):
    if torch.cuda.is_available():
      tracks = self.load('drive/MyDrive/Profesional/AiCore/Projects/MusicGenreClassifier/source/data/tracks.csv')
      features = self.load('drive/MyDrive/Profesional/AiCore/Projects/MusicGenreClassifier/source/data/features.csv')
    else:
      tracks = self.load('source/data/tracks.csv')
      features = self.load('source/data/features.csv')
    mask = tracks['set', 'subset'] <= subset

    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'

    X_train = features.loc[mask & train, 'mfcc']
    y_train = tracks.loc[mask & train, ('track', 'genre_top')]
    X_val = features.loc[mask & val, 'mfcc']
    y_val = tracks.loc[mask & val, ('track', 'genre_top')]
    X_test = features.loc[mask & test, 'mfcc']
    y_test = tracks.loc[mask & test, ('track', 'genre_top')]

    X_train, y_train = self.drop_missing_values(X_train, y_train)
    X_val, y_val = self.drop_missing_values(X_val, y_val)
    X_test, y_test = self.drop_missing_values(X_test, y_test)

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    encoder = LabelEncoder().fit(tracks.loc[:, ('track', 'genre_top')])
    y_train = encoder.transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)

    train_set = ClassData(X_train, y_train, device=self.device)
    val_set = ClassData(X_val, y_val, device=self.device)
    test_set = ClassData(X_test, y_test, device=self.device)

    self.train_load = DataLoader(train_set, batch_size=2**10,
            shuffle=True, num_workers=1)
    self.val_load = DataLoader(val_set, batch_size=2**10,
            shuffle=False, num_workers=1)
    self.test_load = DataLoader(test_set, batch_size=2**10,
            shuffle=False, num_workers=1)

  def drop_missing_values(self, X, y):
    dataset = pd.merge(X, y, left_index=True, right_index=True)
    dataset.dropna(how='any', inplace = True)
    clean_X = dataset[[column for column in X]]
    clean_y = dataset['track', 'genre_top']
    
    return clean_X, clean_y
    
  def load(self, filepath):
      """
      For reference: https://github.com/mdeff/tracks/blob/master/self.py
      """
      filename = os.path.basename(filepath)

      if 'features' in filename:
          return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

      if 'echonest' in filename:
          return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

      if 'genres' in filename:
          return pd.read_csv(filepath, index_col=0)

      if 'tracks' in filename:
          tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

          COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                  ('track', 'genres'), ('track', 'genres_all')]
          for column in COLUMNS:
              tracks[column] = tracks[column].map(ast.literal_eval)

          COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                  ('album', 'date_created'), ('album', 'date_released'),
                  ('artist', 'date_created'), ('artist', 'active_year_begin'),
                  ('artist', 'active_year_end')]
          for column in COLUMNS:
              tracks[column] = pd.to_datetime(tracks[column])

          SUBSETS = ('small', 'medium', 'large')
          try:
              tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                      'category', categories=SUBSETS, ordered=True)
          except (ValueError, TypeError):
              # the categories and ordered arguments were removed in pandas 0.25
              tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                      pd.CategoricalDtype(categories=SUBSETS, ordered=True))

          COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                  ('album', 'type'), ('album', 'information'),
                  ('artist', 'bio')]
          for column in COLUMNS:
              tracks[column] = tracks[column].astype('category')

          return tracks

class ClassData(Dataset):
  """
  Class that implements torch.utils.data.Dataset
  """
  def __init__(self, X, y, device=torch.device("cpu")):
    super().__init__()
    self.device = device

    if len(X.shape) > 1:
      self.n_features = X.shape[1]
      self.X = torch.Tensor(X).reshape(-1, self.n_features).float().to(self.device)
      # self.X = torch.Tensor(X).float()
    else:
      self.n_features = 1
      self.X = torch.Tensor(X.reshape(-1, self.n_features)).float().to(self.device)
      # self.X = torch.Tensor(X.reshape(-1, self.n_features)).float()

    if len(y.shape) > 1:
      self.n_labels = y.shape[1]
      self.y = torch.Tensor(y).long().to(self.device)
    else:
      self.n_labels = 1
      self.y = torch.Tensor(y).reshape(-1, self.n_labels).long().to(self.device)

  def __getitem__(self, idx):
    # X = self.X[idx]
    # X = torch.cat([X, X])
    # return X, self.y[idx]
    return self.X[idx], self.y[idx]

  def __len__(self):
      return len(self.X)

if __name__ == "__main__":
  music = MusicData()
  for x, y in music.train_load:
    print(x.shape, y.shape)
    print(x)
    print(y)
    break