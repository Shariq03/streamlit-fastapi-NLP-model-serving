import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import json


def get_model_and_tokenizer(QNA_model_path):
	if QNA_model_path is None:
		model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
		tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
	else:
		try:
			model = BertForQuestionAnswering.from_pretrained(QNA_model_path)
			tokenizer = BertTokenizer.from_pretrained(QNA_model_path)
		except:
			model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
			tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
	return model, tokenizer

def get_answer_from_text(model, tokenizer, question, answer_text):
	input_ids = tokenizer.encode(question, answer_text)
	tokens = tokenizer.convert_ids_to_tokens(input_ids)
	sep_index = input_ids.index(tokenizer.sep_token_id)
	num_seg_a = sep_index + 1
	num_seg_b = len(input_ids) - num_seg_a
	segment_ids = [0]*num_seg_a + [1]*num_seg_b
	assert len(segment_ids) == len(input_ids)
	# Generate scoring for each token in the input_ids
	l = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
	start_scores, end_scores = l[0], l[1]
	# Getting the final answer
	answer_start = torch.argmax(start_scores)
	answer_end = torch.argmax(end_scores)
	answer = tokens[answer_start]
	for i in range(answer_start + 1, answer_end + 1):
	    if tokens[i][0:2] == '##':
	        answer += tokens[i][2:]	    
	    else:
	        answer += ' ' + tokens[i]
	return answer