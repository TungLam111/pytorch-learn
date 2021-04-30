import json
import csv
import pandas as pd 

def convert_from_json_to_csv_mod():
    f = open('modcloth_final_data.json')
    data = json.load(f)
    f.close()
    json_list = data["data"]
    
    df = pd.DataFrame.from_dict(json_list)
    df.to_csv("mod.csv")

def convert_from_json_csv_rent():
    f = open('renttherunway_final_data.json')
    data = json.load(f)
    f.close()
    json_list = data["data"]
    df = pd.DataFrame.from_dict(json_list)
    df.to_csv("rent.csv")

def clean_csv_rent():
    df = pd.read_csv("rent.csv")
    print(df.head())
    print(df.tail())

def clean_csv_mod():
    df = pd.read_csv("mod.csv")
    print(df.head())
    print(df.tail())
clean_csv_rent()