import numpy as np 
import pandas as pd 
import joblib , json , os , datetime 
from flask import Flask , render_template , request 
from textblob import TextBlob 
import tensorflow as tf 


app = Flask(__name__)


# Location of artifacts
PREP = "data_analysis1/artifacts/preprocess/" 
model_path = "data_analysis1/artifacts/models/"
# Load artifacts 

unique_vals = json.load(open(f"{PREP}/unique_values.json"))
mapped_cities = json.load(open(f"{PREP}/mapped_cities.json"))

CityBinsScaler = joblib.load(f"{PREP}/CityBinsScaler.joblib")
connectedTimebins_scaler = joblib.load(f"{PREP}/connectedTimebins_scaler.joblib")
price_transformer = joblib.load(f"{PREP}/price_transformer.joblib")
price_limits = joblib.load(f"{PREP}/price_limits.joblib")
Remark_scaler = joblib.load(f"{PREP}/remark_word_count_scaler.joblib")
responseTimeScaler = joblib.load(f"{PREP}/responseTimeScaler.joblib")
timeToIssue_scaler = joblib.load(f"{PREP}/timeToIssue_scaler.joblib")


product_ohe = joblib.load(f"{PREP}/product_ohe.joblib")
channel_ohe = joblib.load(f"{PREP}/channel_ohe.joblib")
category_ohe = joblib.load(f"{PREP}/category_ohe.joblib")
sub_category_ohe = joblib.load(f"{PREP}/subcategory_ohe.joblib")
supervisor_ohe = joblib.load(f"{PREP}/supervisor_ohe.joblib")
manager_ohe = joblib.load(f"{PREP}/manager_ohe.joblib")
shift_ohe = joblib.load(f"{PREP}/shift_ohe.joblib")

tenure_oe = joblib.load(f"{PREP}/tenure_oe.joblib")
tenure_scaler = joblib.load(f"{PREP}/tenure_scaler.joblib")
agent_rating = joblib.load(f"{PREP}/agent_rating.joblib")
agent_bin_scaler = joblib.load(f"{PREP}/agent_bins_scaler.joblib")

response_fallback = joblib.load(f"{PREP}/response_time_fallback.joblib")

drop_cols = joblib.load(f"{PREP}/data_matrix_drop_cols.joblib")


ann_model = tf.keras.models.load_model(f"{model_path}/ann5_model.keras")


def remark_sentiment(text):
    return 0 if not text else TextBlob(str(text)).sentiment.polarity 

def remark_wordcount(text):
    return 0 if not text else len(text.split(" "))

def connected_time_bucket(x):
    if x < 250: return 1
    elif x < 500: return 2
    elif x < 750: return 3
    elif x < 1000: return 4
    elif x > 1000: return 5
    return 6

def city_bins(city):
    return mapped_cities.get(city,1)

def time_diff(o,r):
    try:
        o = datetime.datetime.strptime(o,"%d/%m/%Y %H:%M")
        r = datetime.datetime.strptime(r,"%d/%m/%Y %H:%M")

    except:
        return np.random.normal(response_fallback['mean'],response_fallback['std'])
    
    diff = (r-o).total_seconds()/3600
    return diff if diff>0 else np.random.normal(response_fallback['mean'],response_fallback['std'])




@app.route('/')
def index():
    return render_template('index.html',data = unique_vals)


@app.route("/predict", methods = ["POST"])
def predict():
    city = request.form['city']
    product = request.form['product']
    channel = request.form['channel']
    category = request.form['category']
    subcat = request.form['subcategory']
    supervisor = request.form['supervisor']
    manager = request.form['manager']
    shift = request.form['shift']
    tenure = request.form['tenure']
    agent = request.form['agent']

    price = float(request.form['price'])
    remarks = request.form['remarks']
    conn_time = float(request.form['connected_time'])
    order = request.form['order_time']
    issue = request.form['issue_time']
    replied = request.form['reply_time']

    # Preprocessing
    city_binval = city_bins(city)
    city_scaled = CityBinsScaler.transform([[city_binval]])[0][0]
    city_MI = 1 if city is None else 0

    conn_bucket = connected_time_bucket(conn_time)
    conn_scaled = connectedTimebins_scaler.transform([[conn_bucket]])[0][0]
    conn_MI = 1 if conn_time<=0 else 0
    product_MI = 1 if product is None else 0

    ohe_product = product_ohe.transform([[product]])[0]
    ohe_channel = channel_ohe.transform([[channel]])[0]
    ohe_category = category_ohe.transform([[category]])[0]
    ohe_subcat = sub_category_ohe.transform([[subcat]])[0]
    ohe_supervisor = supervisor_ohe.transform([[supervisor]])[0]
    ohe_manager = manager_ohe.transform([[manager]])[0]
    ohe_shift = shift_ohe.transform([[shift]])[0]

    price_norm = price_transformer.transform([[price]])[0][0]

    MI_remarks = 1 if len(remarks.strip())==0 else 0
    senti = remark_sentiment(remarks)
    wc_scaled = Remark_scaler.transform([[remark_wordcount(remarks)]])[0][0]

    hours = time_diff(order,issue)
    hours_scaled = timeToIssue_scaler.transform([[hours]])[0][0]
    resp_hours = time_diff(issue,replied)
    resp_hours_scaled = responseTimeScaler.transform([[resp_hours]])[0][0]

    agen_rate = agent_rating.get(agent,agent_rating.mean())

    agen_scaled = agent_bin_scaler.transform([[agen_rate]])[0][0]

    tenure_num = tenure_oe.transform([[tenure]])[0][0]
    tenure_scaled = tenure_scaler.transform([[tenure_num]])[0][0]

    # Assemble full row 

    row = np.hstack([city_MI , city_scaled , conn_MI , conn_scaled , product_MI, ohe_product ,price_norm ,MI_remarks , senti ,
                     wc_scaled , ohe_channel, ohe_category , ohe_subcat,  hours_scaled , resp_hours_scaled , agen_scaled , ohe_supervisor  , ohe_manager , tenure_scaled  , ohe_shift]).reshape(1,-1)


    # pred = model.predict(row)[0]
    ann_pred_prob = ann_model.predict(row.astype("float32"))[0][0]
    ann_pred_label = 1 if ann_pred_prob>=0.5 else 0

    label_text = "Satisfied" if ann_pred_label==1 else "Not Satisfied"

    return render_template('result.html',prediction=label_text,confidence = round(ann_pred_prob*100 , 2))

app.run(debug = True)