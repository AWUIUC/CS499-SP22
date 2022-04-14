https://dash.plotly.com/deployment

0. create folder for deployed app and cd into the folder
0. create git repo with: git init, and also put the files needed for the app to run in this folder (ex: app.py)
0. add other files app.py needs to run: (Ex: assets folder stuff and COVID_Forecaster_full.csv file)
1. create a virtual env in commandline with: python3 -m venv env
2. Activate evirtual env with: source env/bin/activate
3. install packages: flask, dash, and plotly with pip install flask, pip install dash, and pip install plotly
4. also install gunicorn: pip install gunicorn
5. add .gitignore, Procfile, and requirements.txt 
5. note: when creating requirements.txt, don't just do pip freeze > requirements.txt since that creates strict requirements for dash-table and ends with the following error during deployment: 
remote:        Collecting dash-table
remote:          Downloading dash_table-5.0.0-py3-none-any.whl (3.9 kB)
remote:        ERROR: Could not find a version that satisfies the requirement dataclasses==0.8 (from versions: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
remote:        ERROR: No matching distribution found for dataclasses==0.8
remote:  !     Push rejected, failed to compile Python app.
5. Instead just put the libraries you installed (flask, plotly, dash, gunicorn, and pandas) into the requirements.txt file
6. Run: heroku create covid-forecaster-ui
7. Run: git add .
8. Run: git commit -m "UI code"
9. Run: git push heroku master
10. Run: heroku ps:scale web=1

UI is accessible at: https://covid-forecaster-ui.herokuapp.com/
