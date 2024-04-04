from flask import Flask, make_response,render_template,Response,session,send_file,request,redirect,url_for,flash
import pymysql
from io import BytesIO
import mysql.connector,hashlib
import matplotlib.pyplot as plt
import numpy as np
import re
from fpdf import FPDF
from flask import jsonify
from datetime import datetime
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

mydb = mysql.connector.connect(
  host='localhost',
  user='root',
  password='root123',
  database = 'BI_project_2000_tc'
)
mycursor = mydb.cursor(buffered=True)

app = Flask(__name__)

@app.route('/')
@app.route('/home')   
def home():
    return render_template('home.html')

@app.route('/dashboard')  
def dashboard():
    return render_template('dashboard.html')

@app.route('/res1') 
def res1():
    return render_template('res1.html')

@app.route('/res2')  
def res2():
    return render_template('res2.html')

@app.route('/res3')  
def res3():
    return render_template('res3.html')
    
@app.route('/op1')  
def op1():
    return render_template('op1.html')

@app.route('/op2')  
def op2():
    return render_template('op2.html')

@app.route('/op3')  
def op3():
    return render_template('op3.html')

@app.route('/op4')  
def op4():
    return render_template('op4.html')

@app.route('/vis1')  
def vis1():
    return render_template('vis1.html')

@app.route('/vis2')  
def vis2():
    return render_template('vis2.html')

@app.route('/vis3')  
def vis3():
    return render_template('vis3.html')

@app.route('/vis4')  
def vis4():
    return render_template('vis4.html')

@app.route('/vis5')  
def vis5():
    return render_template('vis5.html')

@app.route('/vis6')  
def vis6():
    return render_template('vis6.html')

@app.route('/vis7') 
def vis7():
    return render_template('vis7.html')

@app.route('/vis8')  
def vis8():
    return render_template('vis8.html')

@app.route('/vis9') 
def vis9():
    return render_template('vis9.html')

@app.route('/vis10') 
def vis10():
    return render_template('vis10.html')

@app.route('/vis11')  
def vis11():
    return render_template('vis11.html')

@app.route('/vis12') 
def vis12():
    return render_template('vis12.html')

@app.route('/vis_board')  
def vis_board():
    return render_template('vis_board.html')




if __name__ == "__main__":
    app.secret_key = 'sec key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
