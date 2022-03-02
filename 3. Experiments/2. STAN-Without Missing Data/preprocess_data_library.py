
"""
Import libraries needed
"""
from data_downloader import GenerateTrainingData
from utils import gravity_law_commute_dist
import pickle
import pandas as pd
import dgl
import numpy as np

"""
Declare global variables used to preprocess data
"""
START_DATE = '2020-05-01'
END_DATE = '2020-12-01'
valid_window = 25
test_window = 25
history_window=6
pred_window=15
slide_step=5

def get_preprocessed_data():
  """
  Download JHU data and merge it with population data
  """
  # Download data
  GenerateTrainingData().download_jhu_data(START_DATE, END_DATE)

  #Merge population data with downloaded data
  raw_data = pickle.load(open('./data/state_covid_data.pickle','rb'))
  pop_data = pd.read_csv('./uszips.csv')
  pop_data = pop_data.groupby('state_name').agg({'population':'sum', 'density':'mean', 'lat':'mean', 'lng':'mean'}).reset_index()
  raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

  """
  Create graph in DGL library based on similarities between locations 
  """
  # Generate location similarity
  loc_list = list(raw_data['state'].unique())
  loc_dist_map = {}
  for each_loc in loc_list:
      loc_dist_map[each_loc] = {}
      for each_loc2 in loc_list:
          lat1 = raw_data[raw_data['state']==each_loc]['latitude'].unique()[0]
          lng1 = raw_data[raw_data['state']==each_loc]['longitude'].unique()[0]
          pop1 = raw_data[raw_data['state']==each_loc]['population'].unique()[0]
          
          lat2 = raw_data[raw_data['state']==each_loc2]['latitude'].unique()[0]
          lng2 = raw_data[raw_data['state']==each_loc2]['longitude'].unique()[0]
          pop2 = raw_data[raw_data['state']==each_loc2]['population'].unique()[0]
          
          loc_dist_map[each_loc][each_loc2] = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5)

  #Generate Graph
  dist_threshold = 18
  for each_loc in loc_dist_map:
      loc_dist_map[each_loc] = {k: v for k, v in sorted(loc_dist_map[each_loc].items(), key=lambda item: item[1], reverse=True)}
  adj_map = {}
  for each_loc in loc_dist_map:
      adj_map[each_loc] = []
      for i, each_loc2 in enumerate(loc_dist_map[each_loc]):
          if loc_dist_map[each_loc][each_loc2] > dist_threshold:
              if i <= 3:
                  adj_map[each_loc].append(each_loc2)
              else:
                  break
          else:
              if i <= 1:
                  adj_map[each_loc].append(each_loc2)
              else:
                  break
  rows = []
  cols = []
  for each_loc in adj_map:
      for each_loc2 in adj_map[each_loc]:
          rows.append(loc_list.index(each_loc))
          cols.append(loc_list.index(each_loc2))

  g = dgl.graph((rows, cols))

  """
  Preprocess data by separating it into different groups
  """
  # Preprocess features
  active_cases = []
  confirmed_cases = []
  new_cases = []
  death_cases = []
  static_feat = []

  for i, each_loc in enumerate(loc_list):
      active_cases.append(raw_data[raw_data['state'] == each_loc]['active'])
      confirmed_cases.append(raw_data[raw_data['state'] == each_loc]['confirmed'])
      new_cases.append(raw_data[raw_data['state'] == each_loc]['new_cases'])
      death_cases.append(raw_data[raw_data['state'] == each_loc]['deaths'])
      static_feat.append(np.array(raw_data[raw_data['state'] == each_loc][['population','density','lng','lat']]))
      
  active_cases = np.array(active_cases)
  confirmed_cases = np.array(confirmed_cases)
  death_cases = np.array(death_cases)
  new_cases = np.array(new_cases)
  static_feat = np.array(static_feat)[:, 0, :]
  recovered_cases = confirmed_cases - active_cases - death_cases
  susceptible_cases = np.expand_dims(static_feat[:, 0], -1) - active_cases - recovered_cases

  # Batch_feat: new_cases(dI), dR, dS
  #dI = np.array(new_cases)
  dI = np.concatenate((np.zeros((active_cases.shape[0],1), dtype=np.float32), np.diff(active_cases)), axis=-1)
  dR = np.concatenate((np.zeros((recovered_cases.shape[0],1), dtype=np.float32), np.diff(recovered_cases)), axis=-1)
  dS = np.concatenate((np.zeros((susceptible_cases.shape[0],1), dtype=np.float32), np.diff(susceptible_cases)), axis=-1)

  """
  Normalize data
  """
  #Build normalizer
  normalizer = {'S':{}, 'I':{}, 'R':{}, 'dS':{}, 'dI':{}, 'dR':{}}

  for i, each_loc in enumerate(loc_list):
      normalizer['S'][each_loc] = (np.mean(susceptible_cases[i]), np.std(susceptible_cases[i]))
      normalizer['I'][each_loc] = (np.mean(active_cases[i]), np.std(active_cases[i]))
      normalizer['R'][each_loc] = (np.mean(recovered_cases[i]), np.std(recovered_cases[i]))
      normalizer['dI'][each_loc] = (np.mean(dI[i]), np.std(dI[i]))
      normalizer['dR'][each_loc] = (np.mean(dR[i]), np.std(dR[i]))
      normalizer['dS'][each_loc] = (np.mean(dS[i]), np.std(dS[i]))

  dynamic_feat = np.concatenate((np.expand_dims(dI, axis=-1), np.expand_dims(dR, axis=-1), np.expand_dims(dS, axis=-1)), axis=-1)
      
  #Normalize
  for i, each_loc in enumerate(loc_list):
      dynamic_feat[i, :, 0] = (dynamic_feat[i, :, 0] - normalizer['dI'][each_loc][0]) / normalizer['dI'][each_loc][1]
      dynamic_feat[i, :, 1] = (dynamic_feat[i, :, 1] - normalizer['dR'][each_loc][0]) / normalizer['dR'][each_loc][1]
      dynamic_feat[i, :, 2] = (dynamic_feat[i, :, 2] - normalizer['dS'][each_loc][0]) / normalizer['dS'][each_loc][1]
  dI_mean = []
  dI_std = []
  dR_mean = []
  dR_std = []
  for i, each_loc in enumerate(loc_list):
      dI_mean.append(normalizer['dI'][each_loc][0])
      dR_mean.append(normalizer['dR'][each_loc][0])
      dI_std.append(normalizer['dI'][each_loc][1])
      dR_std.append(normalizer['dR'][each_loc][1])
  dI_mean = np.array(dI_mean)
  dI_std = np.array(dI_std)
  dR_mean = np.array(dR_mean)
  dR_std = np.array(dR_std)

  """
  Separate data into training, testing, and validation sets
  """
  # Helper function for creating each set of data used
  def prepare_data(data, sum_I, sum_R, history_window=5, pred_window=15, slide_step=5):
      # Data shape n_loc, timestep, n_feat
      # Reshape to n_loc, t, history_window*n_feat
      n_loc = data.shape[0]
      timestep = data.shape[1]
      n_feat = data.shape[2]
      
      x = []
      y_I = []
      y_R = []
      last_I = []
      last_R = []
      concat_I = []
      concat_R = []
      for i in range(0, timestep, slide_step):
          if i+history_window+pred_window-1 >= timestep or i+history_window >= timestep:
              break
          x.append(data[:, i:i+history_window, :].reshape((n_loc, history_window*n_feat)))
          
          concat_I.append(data[:, i+history_window-1, 0])
          concat_R.append(data[:, i+history_window-1, 1])
          last_I.append(sum_I[:, i+history_window-1])
          last_R.append(sum_R[:, i+history_window-1])

          y_I.append(data[:, i+history_window:i+history_window+pred_window, 0])
          y_R.append(data[:, i+history_window:i+history_window+pred_window, 1])

      x = np.array(x, dtype=np.float32).transpose((1, 0, 2))
      last_I = np.array(last_I, dtype=np.float32).transpose((1, 0))
      last_R = np.array(last_R, dtype=np.float32).transpose((1, 0))
      concat_I = np.array(concat_I, dtype=np.float32).transpose((1, 0))
      concat_R = np.array(concat_R, dtype=np.float32).transpose((1, 0))
      y_I = np.array(y_I, dtype=np.float32).transpose((1, 0, 2))
      y_R = np.array(y_R, dtype=np.float32).transpose((1, 0, 2))
      return x, last_I, last_R, concat_I, concat_R, y_I, y_R

  #Split train-test
  train_feat = dynamic_feat[:, :-valid_window-test_window, :]
  val_feat = dynamic_feat[:, -valid_window-test_window:-test_window, :]
  test_feat = dynamic_feat[:, -test_window:, :]

  train_x, train_I, train_R, train_cI, train_cR, train_yI, train_yR = prepare_data(train_feat, active_cases[:, :-valid_window-test_window], recovered_cases[:, :-valid_window-test_window], history_window, pred_window, slide_step)
  val_x, val_I, val_R, val_cI, val_cR, val_yI, val_yR = prepare_data(val_feat, active_cases[:, -valid_window-test_window:-test_window], recovered_cases[:, -valid_window-test_window:-test_window], history_window, pred_window, slide_step)
  test_x, test_I, test_R, test_cI, test_cR, test_yI, test_yR = prepare_data(test_feat, active_cases[:, -test_window:], recovered_cases[:, -test_window:], history_window, pred_window, slide_step)

  """
  Package/organize preprocessed data together into a dictionary called "preprocessed_data"
  """
  training_variables = {'train_x':train_x, 'train_I':train_I, 'train_R':train_R, 
                        'train_cI':train_cI, 'train_cR':train_cR, 
                        'train_yI':train_yI, 'train_yR':train_yR}
  validation_variables = {'val_x':val_x, 'val_I':val_I, 'val_R':val_R, 
                          'val_cI':val_cI, 'val_cR':val_cR,
                          'val_yI':val_yI, 'val_yR':val_yR}
  testing_variables = {'test_x':test_x, 'test_I':test_I, 'test_R':test_R, 
                      'test_cI':test_cI, 'test_cR':test_cR,
                      'test_yI':test_yI, 'test_yR':test_yR}
  normalization_variables = {'dI_mean':dI_mean, 'dI_std':dI_std, 'dR_mean':dR_mean, 'dR_std':dR_std}

  preprocessed_data = {
      'training_variables':training_variables,
      'validation_variables':validation_variables,
      'testing_variables':testing_variables,
      'normalization_variables':normalization_variables,
      'static_feat':static_feat,
      'loc_list':loc_list,
      'graph':g,
      'active_cases':active_cases
  }

  return preprocessed_data