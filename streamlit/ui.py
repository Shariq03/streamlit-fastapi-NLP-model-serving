import streamlit as st
import requests
backend = "http://fastapi:8000/run_QNA"

st.title("BERT based QNA System")

st.write(
	"""
	Obtain answer from the given text and a given question via BERT implemented in Pytorch.
	"""
)

text = st.text_area(label="Your Text here", value="", height = 200)
question = st.text_input("Your Question here", "")

if st.button("Get Answer"):
	if (question == "") | (text == ""):
		st.write("You need to input some question and text")
	else:
		try:
			params = {"question":question, "text":text}
			response = requests.get(url=backend, json=params)
			if response.status_code == 200:
				answer = response.text
				st.write(answer.title())
			else:
				st.write("Request to server Failed!")
		except:
			st.write("An exception occured!")


