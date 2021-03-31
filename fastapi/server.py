import warnings
warnings.filterwarnings('ignore')
from bert_QNA import get_model_and_tokenizer, get_answer_from_text
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()


model_path = "./bert-large-uncased-whole-word-masking-finetuned-squad"
model, tokenizer = get_model_and_tokenizer(model_path)


class QNA(BaseModel):
    question: str
    text: str


@app.get("/")
def hello_world():
	return "Hello World!!!"


@app.get("/run_QNA")
def run_QNA(qna : QNA):
	question = qna.question
	text = qna.text
	answer = get_answer_from_text(model, tokenizer, question, text)
	print (question, text, answer)
	return answer