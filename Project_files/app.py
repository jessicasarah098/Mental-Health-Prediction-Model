from flask import Flask, render_template, request,jsonify
import pickle, joblib 
import pandas as pd

app = Flask (__name__)
#model = pickle.load(open("model.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

ct = joblib.load('feature_values')

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/index')
def predict():
    return render_template("index.html")

@app.route('/out', methods =["POST"]) 
def output():
    print("heyyy")
   # data = request.json
   # print(request.json)
    # age = data["age"]
 
    # gender = data["gender"]
    # self_employed = data["self_employed"].lower()
    # family_history = data["family_history"].lower()
    # work_interfere = data["work_interfere"].lower()
    # no_employees = data["no_employees"]
    # remote_work = data["remote_work"].lower()
    # tech_company = data["tech_company"].lower()
    # benefits = data["benefits"].lower()
    # care_options = data["care_options"].lower()
    # wellness_program = data["wellness_program"].lower()
    # seek_help = data["seek_help"].lower()
    # anonymity = data["anonymity"].lower()
    # leave = data["leave"].lower()
    # mental_health_consequence = data["mental_health_consequence"].lower()
    # phys_health_consequence = data ["phys_health_consequence"].lower()
    # coworkers = data[ "coworkers"].lower()
    # supervisor = data["supervisor"].lower()
    # mental_health_interview = data ["mental_health_interview"].lower()
    # phys_health_interview = data["phys_health_interview"].lower()
    # mental_vs_physical = data["mental_vs_physical"].lower() 
    # obs_consequence = data["obs_consequence"].lower()

    age = request.form["age"]
    gender = request.form["gender"]
    self_employed = request.form["self_employed"].lower()
    family_history = request.form["family_history"].lower()
    work_interfere = request.form["work_interfere"].lower()
    #no_employees = request.form["no_employees"]
    remote_work = request.form["remotely"].lower()
    tech_company = request.form["employer"].lower()
    benefits = request.form["benefits"].lower()
    care_options = request.form["care_options"].lower()
    wellness_program = request.form["wellness_program"].lower()
    seek_help = request.form["seek_help"].lower()
    anonymity = request.form["aNonymity"].lower()
    leave = request.form["Leave"].lower()
    mental_health_consequence = request.form["mental_health_consequence"].lower()
    phys_health_consequence = request.form ["health_consequence"].lower()
    coworkers = request.form[ "coworkers"].lower()
    supervisor = request.form["supervisor"].lower()
    mental_health_interview = request.form ["mental_health_interview"].lower()
    phys_health_interview = request.form["phys_health_interview"].lower()
    mental_vs_physical = request.form["mental_vS_physical"].lower() 
    obs_consequence = request.form["obs_consequence"].lower()


    
    data = [[age, gender, self_employed, family_history, work_interfere, remote_work, tech_company, benefits, care_options,
        wellness_program, seek_help, anonymity, leave, mental_health_consequence, phys_health_consequence, coworkers, supervisor,
        mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence]]
    feature_cols = ['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work', 'tech_company',
                    'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']
    
    print(data)
    print(model)
    #print(pd.DataFrame(data,columns=feature_cols))
    #print(ct.transform(pd.DataFrame(data,columns=feature_cols)))
    
    pred = model.predict(ct.transform(pd.DataFrame(data, columns=feature_cols)))
    pred = pred[0] 
    if pred == 1:
        return render_template("output.html",y="The person requires mental health treatment")
    else:
        return render_template("output.html",y="The person doesn't requires mental health treatment")
    return jsonify({"data":str(pred)})
    

if __name__ ==  '__main__':
    app.run(debug = True)