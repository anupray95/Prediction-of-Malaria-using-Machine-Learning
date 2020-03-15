from flask import Flask, render_template, request
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/',methods = ['POST'])
def getvalue():

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_excel('a.xlsx')
    
    df.set_index('Year')

    # Global variables
    patients = 0
    max_percent = 0


    #value from html

    dist = request.form['dist']
    yr = request.form['year']
    if yr == '':
        yr = 0
    else:
        yr = int(yr)


    #mycode

   
    if dist == '':


        #Select year as a variable
        
        df1 = df.loc[df['Year'] == yr]
        
        
        # Print the maximum percentage
        max_percent = df1['Percent'].max()
        print(max_percent)

        b = df.loc[df['Percent'] ==max_percent ][['District','MalariaPatient']]


        # convert dataframe b  to numpy array for to ease operations
        a = b.to_numpy()

        #print the district having maximum percentage
        dist =a[0][0]
        print(dist)

        #print total suffered patient
        patients = a[0][1]
        print(patients)
        





    
    if yr == 0:
        
            
        #Select district as a variable
       
        df1 = df.loc[df['District'] == dist]

        # Print the maximum percentage
        max_percent = df1['Percent'].max()
        print(max_percent)

        b = df.loc[df['Percent'] == max_percent][['Year','MalariaPatient']]

        # convert dataframe b  to numpy array for to ease operations
        a = b.to_numpy()

        #print the district having maximum percentage
        yr =a[0][0]
        print(yr)
        
        #print total suffered patient
        patients = a[0][1]
        print(patients)
    
    
    #Applying Regression Model
    X = df['Year']

    y = df['MalariaPatient']
    X = np.array(X)

    X.reshape(-1, 1)
    X = pd.DataFrame(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state = 101)
    from sklearn.linear_model import LinearRegression
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    coef = int(lm.coef_)

    return render_template('pass.html',patients = patients, max_percent = max_percent, dist = dist, yr = yr, coef = coef)


    
    
    
if __name__ == '__main__':
    app.run(debug=True)