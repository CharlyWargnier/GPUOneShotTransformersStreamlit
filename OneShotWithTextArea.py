import pandas as pd
import numpy as np
from transformers import pipeline
import streamlit as st



st.title("One Shot Classifier")

##########################


c30, c31 = st.beta_columns(2)

with c30:
	with st.beta_expander("Bucket 01", expanded=False):
		st.write("boots")
		st.write("car")
		st.write("boat")

with c31:
	with st.beta_expander("Bucket 02", expanded=False):
		st.write("heels")
		st.write("SUV")
		st.write("ferry")


###################

c32, c33 = st.beta_columns(2)

with c32:
	with st.beta_expander("candidate_labels 01", expanded=False):
		st.write("shoe")
		st.write("sea")
		st.write("automotive")

with c33:
	with st.beta_expander("candidate_labels 02", expanded=False):
		st.write("shoes")
		st.write("beaches")
		st.write("cars")

###################


MAX_LINES = 10

text = st.text_area("keyword, one per line.", height=150)
lines = text.split("\n")  # A list of lines

if len(lines) > MAX_LINES:
    st.warning(f"Maximum number of lines reached. Only the first {MAX_LINES} will be processed.")
    lines = lines[:MAX_LINES]

for line in lines:
    data = pd.DataFrame({'url':lines})

if not text:
    st.warning('Paste some keywords.')
    st.stop()

datalist = data.url.tolist()

datalistDeduped = [] 
Variable = [datalistDeduped.append(x) for x in datalist if x not in datalistDeduped] 

blocklist = {'',' ','-'}

candidate_labels = [x for x in datalistDeduped if x not in blocklist]

st.write("candidate_labels")
st.write(candidate_labels)
#candidate_labels = ["shoe", "sea", "automotive"]

#st.stop()

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModel.from_pretrained("facebook/bart-large-mnli")

##########################

classifier = pipeline("zero-shot-classification")
#classifier = pipeline("zero-shot-classification", device=0) # to utilize GPU

df2 = pd.DataFrame(np.array([["boots", 2,3 ], ["boat", 5, 6], ["car", 8, 9]]),columns=['keyword', 'b', 'c'])

st.write(df2)

sequence = df2['keyword']

#candidate_labels = ["shoe", "sea"]

results = classifier(sequence, candidate_labels)

#st.write(results)

dfnew = pd.DataFrame(results)
st.write(dfnew)

st.stop()


