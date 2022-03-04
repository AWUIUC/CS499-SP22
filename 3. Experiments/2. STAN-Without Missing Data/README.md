Differences between 1. STAN-Original and 2. STAN-Without Missing Data: 

| Original File(s)                                                                 | New File(s)                                                                            | Description of changes                                                                        | Notes about changes |
| --------------------                                                             | ----------------------------------------------------------------------------------     | --------------------------------------------------------------------------------------------- | ------------------- |
| 1. STAN-Original (all files in different directory)                              | preprocess_data.ipynb/preprocess_data_library.py, train_stan.ipynb (current directory) | Separated data preprocessing from training pipeline for ease of reading code + making changes |                     |
| preprocess_data.ipynb/preprocess_data_library.py, model.py, train_stan.ipynb     | preprocess_data2.ipynb/preprocess_data_library2.py, model2.py, trian_stan2.ipynb       | Removed use of unreported data (and consequently removed use of SIR model)                    | Predictions are no longer negative but magnitude of predictions is inaccurate |
| preprocess_data2.ipynb/preprocess_data_library2.py, train_stan2.ipynb            | preprocess_data3.ipynb/preprocess_data_library3.py, train_stan3.ipynb                  | Tried to improve accuracy by smoothing data --> not big enough change to make difference      | model is same as model2.py |
| train_stan2.ipynb, model2.py                                                     | train_stan4.ipynb, model4.py                                                           | Tried to improve accuracy by only predicting # confirmed cases (in case backprop for # deaths was affecting backprop for # confirmed cases) | Changed # linear layers from 2 to 1 to test changes, didn't have much effect, but afterwards tried increasing # epochs from 50 to 200 which showed much greater change in results, albeit still not large enough of change |

3/2/22:
- Separated data preprocessing from training pipeline (reflected in: preprocess_data.ipynb/preprocess_data_library.py, and train_stan.ipynb)
- Removed use of unreported data and removed normalization from preprocessing (reflected in: preprocess_data2.ipynb/preprocess_data_library2.py)
- Removed use of unreported data (and consequently removed SIR model) in model (reflected in: model2.py and train_stan2.ipynb)

3/3/22:
- Tried to improve predictions for confirmed cases by smoothing data (reflected by preprocess_data3.ipynb/preprocess_data_library3.py and train_stan3.ipynb)
- Seems like predictions for total cumulative confirmed case numbers and predictions for deaths are on same order of magnitude (between 0 - 100) 
for both without and with smoothing which makes me suspect whether the architecture (rather than preprocessing) 
is the reason behind why the model isn't predicting/performing well in predicting confirmed case numbers
- Tried changing output from 2 linear layers to using only 1 linear layer to predict only 1 thing (# confirmed cases) 
as opposed to 2 things (# confirmed cases & # deaths), since calculation from 2 losses with different magnitudes might be reason why model isn't performing well 
- ^^ didn't make much difference 
- Tried increasing # epochs --> made difference but will take long time to train to get magnitude of output to be in line with what is expected

Potential things to try:
- don't train a separate model for each state (use the same model/GNN for all states to capture the essence of GNN and also to decrease training time)
- predict change in # cases, and change in # deaths instead of cumulative total # cases/deaths (less of a range for error)
OR change model to predict change in # cases and add that to # total cases at start of timestep so we can predict # total cases
