Differences between 1. STAN-Original and 2. STAN-Without Missing Data: 

3/2/22:
- Separated data preprocessing from training pipeline (reflected in: preprocess_data.ipynb/preprocess_data_library.py, and train_stan.ipynb)
- Removed use of unreported data and removed normalization from preprocessing (reflected in: preprocess_data2.ipynb/preprocess_data_library2.py)
- Removed use of unreported data (and consequently removed SIR model) in model (reflected in: model2.py and train_stan2.ipynb)

3/3/22:
- Tried to improve predictions for confirmed cases by smoothing data (reflected by preprocess_data3.ipynb/preprocess_data_library3.py and train_stan3.ipynb)
- Seems like predictions for total cumulative confirmed case numbers and predictions for deaths are on same order of magnitude (between 0 - 100) 
for both without and with smoothing which makes me suspect whether the architecture (rather than preprocessing) 
is the reason behind why the model isn't predicting/performing well in predicting confirmed case numbers

Potential things to try:
- concatenate the 2 end linear layers together so only 1 loss calculation is needed 
    (perhaps the backprop calculation from 2 losses with vastly different magnitudes is the reason why the model isn't performing well)
- remove 1 linear layer to only predict 1 thing (# confirmed cases, or # deaths) as opposed to 2 things, since # deaths seems like it isn't well reported
- predict change in # cases, and change in # deaths instead of cumulative total # cases/deaths (less of a range for error)