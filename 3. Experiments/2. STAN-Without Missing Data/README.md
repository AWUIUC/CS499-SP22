Differences between 1. STAN-Original and 2. STAN-Without Missing Data: 
Separated data preprocessing from training pipeline (reflected in: preprocess_data.ipynb/preprocess_data_library.py, and train_stan.ipynb)
Removed use of unreported data and removed normalization from preprocessing (reflected in: preprocess_data2.ipynb/preprocess_data_library2.py)
Removed use of unreported data (and consequently removed SIR model) in model (reflected in: model2.py and train_stan2.ipynb)