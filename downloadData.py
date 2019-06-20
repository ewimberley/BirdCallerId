#!/usr/bin/python3
import pandas as pd
import wget
import os

dataFile = "data.csv"
urlPrefix = "https://download.ams.birds.cornell.edu/api/v1/asset/"
pathPrefix = "Data\\"
df = pd.read_csv(dataFile, sep="\t")
for index, row in df.iterrows():
    dataFile = str(row['id']) + ".mp3"
    exists = os.path.isfile(pathPrefix + dataFile)
    if not exists:
        url = urlPrefix + dataFile
        print(url)
        wget.download(url, pathPrefix + dataFile)