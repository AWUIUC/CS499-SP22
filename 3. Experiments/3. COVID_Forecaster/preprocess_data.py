
"""
Import libraries needed
"""
from data_downloader import GenerateTrainingData
from utils import gravity_law_commute_dist
import pickle
import pandas as pd
import numpy as np
import torch

"""
Declare global variables used to preprocess data
"""
START_DATE = '2020-04-12'
END_DATE = '2022-01-24'
valid_window = 25
test_window = 25
history_window=6
pred_window=15
slide_step=5

def get_preprocessed_data():
    """
    Download JHU data
    """
    # Download data
    GenerateTrainingData().download_jhu_data(START_DATE, END_DATE)

    #Merge population data with downloaded data
    raw_data = pickle.load(open('./data/state_covid_data.pickle','rb'))
    pop_data = pd.read_csv('./uszips.csv')
    pop_data = pop_data.groupby('state_name').agg({'population':'sum', 'density':'mean', 'lat':'mean', 'lng':'mean'}).reset_index()
    raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

    #############################################################################################################################################################################

    """
    Create edge index to be passed to GNN architecture later in Pytorch Geometric
    """
    # State name to state abbreviation mapping (so we can index the state adjacency map later)
    # Reference: https://gist.github.com/rogerallen/1583593 
    us_state_to_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY",
        "District of Columbia": "DC",
        "American Samoa": "AS",
        "Guam": "GU",
        "Northern Mariana Islands": "MP",
        "Puerto Rico": "PR",
        "United States Minor Outlying Islands": "UM",
        "U.S. Virgin Islands": "VI",
    }

    # invert the dictionary
    abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

    # State abbreviation to state adjacency list mapping (for creation of map)
    # Modified from: https://gist.github.com/rietta/4112447 
    states_adjacency_list = {
        "AK": "AK",
        "AL": "AL,MS,TN,GA,FL",
        "AR": "AR,MO,TN,MS,LA,TX,OK",
        "AZ": "AZ,CA,NV,UT,CO,NM",
        "CA": "CA,OR,NV,AZ",
        "CO": "CO,WY,NE,KS,OK,NM,AZ,UT",
        "CT": "CT,NY,MA,RI",
        "DC": "DC,MD,VA",
        "DE": "DE,MD,PA,NJ",
        "FL": "FL,AL,GA",
        "GA": "GA,FL,AL,TN,NC,SC",
        "HI": "HI",
        "IA": "IA,MN,WI,IL,MO,NE,SD",
        "ID": "ID,MT,WY,UT,NV,OR,WA",
        "IL": "IL,IN,KY,MO,IA,WI",
        "IN": "IN,MI,OH,KY,IL",
        "KS": "KS,NE,MO,OK,CO",
        "KY": "KY,IN,OH,WV,VA,TN,MO,IL",
        "LA": "LA,TX,AR,MS",
        "MA": "MA,RI,CT,NY,NH,VT",
        "MD": "MD,VA,WV,PA,DC,DE",
        "ME": "ME,NH",
        "MI": "MI,WI,IN,OH",
        "MN": "MN,WI,IA,SD,ND",
        "MO": "MO,IA,IL,KY,TN,AR,OK,KS,NE",
        "MS": "MS,LA,AR,TN,AL",
        "MT": "MT,ND,SD,WY,ID",
        "NC": "NC,VA,TN,GA,SC",
        "ND": "ND,MN,SD,MT",
        "NE": "NE,SD,IA,MO,KS,CO,WY",
        "NH": "NH,VT,ME,MA",
        "NJ": "NJ,DE,PA,NY",
        "NM": "NM,AZ,UT,CO,OK,TX",
        "NV": "NV,ID,UT,AZ,CA,OR",
        "NY": "NY,NJ,PA,VT,MA,CT",
        "OH": "OH,PA,WV,KY,IN,MI",
        "OK": "OK,KS,MO,AR,TX,NM,CO",
        "OR": "OR,CA,NV,ID,WA",
        "PA": "PA,NY,NJ,DE,MD,WV,OH",
        "PR": "PR",
        "RI": "RI,CT,MA",
        "SC": "SC,GA,NC",
        "SD": "SD,ND,MN,IA,NE,WY,MT",
        "TN": "TN,KY,VA,NC,GA,AL,MS,AR,MO",
        "TX": "TX,NM,OK,AR,LA",
        "UT": "UT,ID,WY,CO,NM,AZ,NV",
        "VA": "VA,NC,TN,KY,WV,MD,DC",
        "VT": "VT,NY,NH,MA",
        "WA": "WA,ID,OR",
        "WI": "WI,MI,MN,IA,IL",
        "WV": "WV,OH,PA,MD,VA,KY",
        "WY": "WY,MT,SD,NE,CO,UT,ID"
    }


    # we will use undirected graph, where nodes are represented by ints
    edge_list_source_node = []
    edge_list_destination_node = []


    state_list = list(raw_data['state'].unique())
    for state_name in state_list:
      state_abbrev = us_state_to_abbrev[state_name]
      curr_state_and_neighbors = states_adjacency_list[state_abbrev]
      comma_delimited_list = curr_state_and_neighbors.split(",")
      
      source_state_abbrev = None
      dest_state_abbreviations = None
      if len(comma_delimited_list) == 1:
        source_state_abbrev = comma_delimited_list[0]
        dest_state_abbreviations = [comma_delimited_list[0]]
      else:
        source_state_abbrev = comma_delimited_list[0]
        dest_state_abbreviations = comma_delimited_list[1:]
      
      for dest_state_abbrev in dest_state_abbreviations:
        source_state_full_name = abbrev_to_us_state[source_state_abbrev]
        dest_state_full_name = abbrev_to_us_state[dest_state_abbrev]

        source_state_int = state_list.index(source_state_full_name)
        dest_state_int = state_list.index(dest_state_full_name)
        
        edge_list_source_node.append(source_state_int)
        edge_list_destination_node.append(dest_state_int)

    edge_index = torch.tensor([edge_list_source_node,
                              edge_list_destination_node], dtype=torch.long)

    #############################################################################################################################################################################

    """
    Preprocess data by separating it into different groups
    """
    # Preprocess features
    confirmed_cases = []
    death_cases = []
    static_feat = []

    for i, each_loc in enumerate(state_list):
        confirmed_cases.append(raw_data[raw_data['state'] == each_loc]['confirmed'])
        death_cases.append(raw_data[raw_data['state'] == each_loc]['deaths'])
        static_feat.append(np.array(raw_data[raw_data['state'] == each_loc][['population','density','lng','lat']]))
        
    confirmed_cases_unsmoothed = np.array(confirmed_cases)
    death_cases_unsmoothed = np.array(death_cases)
    static_feat_unsmoothed = np.array(static_feat)[:, 0, :]


    # Calculate change in # cases and # deaths from previous day
    daily_change_in_confirmed_unsmoothed = np.concatenate((np.zeros((confirmed_cases_unsmoothed.shape[0], 1), dtype=np.float32), np.diff(confirmed_cases_unsmoothed)), axis=-1)
    daily_change_in_deaths_unsmoothed = np.concatenate((np.zeros((death_cases_unsmoothed.shape[0], 1), dtype=np.float32), np.diff(death_cases_unsmoothed)), axis=-1)

    #############################################################################################################################################################################

    """
    Smooth the data
    """

    confirmed_cases_smoothed = []
    death_cases_smoothed = []
    daily_change_in_confirmed_smoothed = []
    daily_change_in_deaths_smoothed = []

    # Define smoothing function from: https://www.delftstack.com/howto/python/smooth-data-in-python/
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    for i in range(confirmed_cases_unsmoothed.shape[0]):
      confirmed_cases_smoothed.append(smooth(confirmed_cases_unsmoothed[i], 8))
      death_cases_smoothed.append(smooth(death_cases_unsmoothed[i], 8))
      daily_change_in_confirmed_smoothed.append(smooth(daily_change_in_confirmed_unsmoothed[i], 8))
      daily_change_in_deaths_smoothed.append(smooth(daily_change_in_deaths_unsmoothed[i], 8))

    confirmed_cases_smoothed = np.array(confirmed_cases_smoothed)
    death_cases_smoothed = np.array(death_cases_smoothed)
    daily_change_in_confirmed_smoothed = np.array(daily_change_in_confirmed_smoothed)
    daily_change_in_deaths_smoothed = np.array(daily_change_in_deaths_smoothed)

    #############################################################################################################################################################################

    """
    Put data together into 1 big numpy array
    """
    dynamic_feat_unsmoothed = np.concatenate((np.expand_dims(confirmed_cases_unsmoothed, axis=-1),
                                  np.expand_dims(death_cases_unsmoothed, axis=-1),
                                  np.expand_dims(daily_change_in_confirmed_unsmoothed, axis=-1), 
                                  np.expand_dims(daily_change_in_deaths_unsmoothed, axis=-1)
                                  ), axis=-1)

    dynamic_feat_smoothed = np.concatenate((np.expand_dims(confirmed_cases_smoothed, axis=-1),
                                  np.expand_dims(death_cases_smoothed, axis=-1),
                                  np.expand_dims(daily_change_in_confirmed_smoothed, axis=-1), 
                                  np.expand_dims(daily_change_in_deaths_smoothed, axis=-1)
                                  ), axis=-1)

    #############################################################################################################################################################################

    """
    Separate data into training, testing, and validation sets
    """

    #Split train-test
    train_feat_unsmoothed = dynamic_feat_unsmoothed[:, :-valid_window-test_window, :]
    val_feat_unsmoothed = dynamic_feat_unsmoothed[:, -valid_window-test_window:-test_window, :]
    test_feat_unsmoothed = dynamic_feat_unsmoothed[:, -test_window:, :]

    train_feat_smoothed = dynamic_feat_smoothed[:, :-valid_window-test_window, :]
    val_feat_smoothed = dynamic_feat_smoothed[:, -valid_window-test_window:-test_window, :]
    test_feat_smoothed = dynamic_feat_smoothed[:, -test_window:, :]

    # Helper function for creating each set of data used
    def prepare_data(data):
      # Data shape num_locations, timestep, n_feat
      num_locations = data.shape[0]
      timestep = data.shape[1]
      n_feat = data.shape[2]

      input_entries = []
      output_entries_confirmed = []
      output_entries_deaths = []
      output_entries_change_in_confirmed = []
      output_entries_change_in_deaths = []

      for i in range(0, timestep, slide_step):
        if i+history_window+pred_window-1 >= timestep or i+history_window >= timestep:
            break

        # Shape = number nodes x num_input_features
        input_entry = data[:, i:i+history_window, :].reshape((num_locations, history_window*n_feat)).tolist()

        # Shape = number nodes x num_output_features
        output_entry_confirmed = data[:, i+history_window:i+history_window+pred_window, 0].reshape((num_locations, pred_window)).tolist()
        output_entry_deaths = data[:, i+history_window:i+history_window+pred_window, 1].reshape((num_locations, pred_window)).tolist()
        output_entry_change_in_confirmed = data[:, i+history_window:i+history_window+pred_window, 2].reshape((num_locations, pred_window)).tolist()
        output_entry_change_in_deaths = data[:, i+history_window:i+history_window+pred_window, 3].reshape((num_locations, pred_window)).tolist()

        input_entries.append(torch.tensor(input_entry))
        output_entries_confirmed.append(torch.tensor(output_entry_confirmed))
        output_entries_deaths.append(torch.tensor(output_entry_deaths))
        output_entries_change_in_confirmed.append(torch.tensor(output_entry_change_in_confirmed))
        output_entries_change_in_deaths.append(torch.tensor(output_entry_change_in_deaths))

      return input_entries, output_entries_confirmed, output_entries_deaths, output_entries_change_in_confirmed, output_entries_change_in_deaths

    train_x_unsmoothed, train_y_confirmed_unsmoothed, train_y_deaths_unsmoothed, train_y_change_in_confirmed_unsmoothed, train_y_change_in_deaths_unsmoothed = prepare_data(train_feat_unsmoothed)
    val_x_unsmoothed, val_y_confirmed_unsmoothed, val_y_deaths_unsmoothed, val_y_change_in_confirmed_unsmoothed, val_y_change_in_deaths_unsmoothed = prepare_data(val_feat_unsmoothed)
    test_x_unsmoothed, test_y_confirmed_unsmoothed, test_y_deaths_unsmoothed, test_y_change_in_confirmed_unsmoothed, test_y_change_in_deaths_unsmoothed = prepare_data(test_feat_unsmoothed)

    train_x_smoothed, train_y_confirmed_smoothed, train_y_deaths_smoothed, train_y_change_in_confirmed_smoothed, train_y_change_in_deaths_smoothed = prepare_data(train_feat_smoothed)
    val_x_smoothed, val_y_confirmed_smoothed, val_y_deaths_smoothed, val_y_change_in_confirmed_smoothed, val_y_change_in_deaths_smoothed = prepare_data(val_feat_smoothed)
    test_x_smoothed, test_y_confirmed_smoothed, test_y_deaths_smoothed, test_y_change_in_confirmed_smoothed, test_y_change_in_deaths_smoothed = prepare_data(test_feat_smoothed)

    #############################################################################################################################################################################

    """
    Package/organize preprocessed data together into a dictionary called "preprocessed_data"
    """
    training_variables = {'train_x_unsmoothed':train_x_unsmoothed,
                          'train_x_smoothed':train_x_smoothed, 
                          'train_y_confirmed_unsmoothed':train_y_confirmed_unsmoothed,
                          'train_y_confirmed_smoothed':train_y_confirmed_smoothed,
                          'train_y_deaths_unsmoothed':train_y_deaths_unsmoothed,
                          'train_y_deaths_smoothed':train_y_deaths_smoothed,
                          'train_y_change_in_confirmed_unsmoothed':train_y_change_in_confirmed_unsmoothed,
                          'train_y_change_in_confirmed_smoothed':train_y_change_in_confirmed_smoothed,
                          'train_y_change_in_deaths_unsmoothed':train_y_change_in_deaths_unsmoothed,
                          'train_y_change_in_deaths_smoothed':train_y_change_in_deaths_smoothed
                          }

    validation_variables = {'val_x_unsmoothed':val_x_unsmoothed,
                            'val_x_smoothed':val_x_smoothed,
                            'val_y_confirmed_unsmoothed':val_y_confirmed_unsmoothed,
                            'val_y_confirmed_smoothed':val_y_confirmed_smoothed,
                            'val_y_deaths_unsmoothed':val_y_deaths_unsmoothed,
                            'val_y_deaths_smoothed':val_y_deaths_smoothed,
                            'val_y_change_in_confirmed_unsmoothed':val_y_change_in_confirmed_unsmoothed,
                            'val_y_change_in_confirmed_smoothed':val_y_change_in_confirmed_smoothed,
                            'val_y_change_in_deaths_unsmoothed':val_y_change_in_deaths_unsmoothed,
                            'val_y_change_in_deaths_smoothed':val_y_change_in_deaths_smoothed
                            }

    testing_variables = {'test_x_unsmoothed':test_x_unsmoothed,
                        'test_x_smoothed':test_x_smoothed, 
                        'test_y_confirmed_unsmoothed':test_y_confirmed_unsmoothed,
                        'test_y_confirmed_smoothed':test_y_confirmed_smoothed,
                        'test_y_deaths_unsmoothed':test_y_deaths_unsmoothed,
                        'test_y_deaths_smoothed':test_y_deaths_smoothed,
                        'test_y_change_in_confirmed_unsmoothed':test_y_change_in_confirmed_unsmoothed,
                        'test_y_change_in_confirmed_smoothed':test_y_change_in_confirmed_smoothed,
                        'test_y_change_in_deaths_unsmoothed':test_y_change_in_deaths_unsmoothed,
                        'test_y_change_in_deaths_smoothed':test_y_change_in_deaths_smoothed
                        }

    preprocessed_data = {
        'training_variables':training_variables,
        'validation_variables':validation_variables,
        'testing_variables':testing_variables,
        'edge_index':edge_index
    }

    return preprocessed_data