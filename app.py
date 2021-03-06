import pandas as pd
import numpy as np
from transformers import pipeline

import streamlit as st

##########################

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModel.from_pretrained("facebook/bart-large-mnli")

##########################


st.header("0")

st.stop()

classifier = pipeline("zero-shot-classification")
#classifier = pipeline("zero-shot-classification", device=0) # to utilize GPU

df2 = pd.DataFrame(np.array([["boots", 2,3 ], ["boat", 5, 6], ["car", 8, 9]]),columns=['keyword', 'b', 'c'])

st.write(df2)

sequence = df2['keyword']

candidate_labels = ["shoe", "sea", "automotive"]

results = classifier(sequence, candidate_labels)

dfnew = pd.DataFrame(results)

st.write(dfnew)
