import io
import numpy
import pickle
from sys import argv

def train_data(data):
	lines = data.readlines()
	data_words = []
	for line in lines:
		if line.startswith("#") or line.startswith("\n"):
			continue
		line_split = line.split("\t")
		data_words.append(line_split[0:2])

	del(lines)
	sentences = []
	sentence = []
	W = len(data_words)
	word_types = dict()
	word_types["<UNK>"] = 0

	#Changing first occurrence types to unknown word
	for word in data_words:
		if word[0] is '1':
			sentences.append(sentence)
			sentence = []
		token = word[1]
		if token not in word_types.keys():

			word_types[token] = 0
			word_types["<UNK>"] += 1
			sentence.append("<UNK>")
		else:
			word_types[token] += 1
			sentence.append(token)

	del(sentence)
	del(sentences[0])

	words = dict()

	for i,j in word_types.items():
		if j is not 0:
			words[i] = j
	del(word_types)

	V = len(words)

	pairs = dict()
	pairs_count = 0
	for sentence in sentences:
		length_sentence = len(sentence)
		pairs_count += length_sentence + 1 
		pair = ("<BOS>",sentence[0])
		if pair in pairs.keys():
			pairs[pair] += 1
		else:
			pairs[pair] = 1
		
		for i in range(0,length_sentence-1):
			pair = (sentence[i],sentence[i+1])
			if pair in pairs.keys():
				pairs[pair] += 1
			else:
				pairs[pair] = 1
		
		pair = (sentence[length_sentence-1],"<EOS>")
		if pair in pairs.keys():
			pairs[pair] += 1
		else:
			pairs[pair] = 1

	return W, words, pairs, len(sentences)

def unigram_mle(W,t_words,ft,dev_file):
	#setting frequency threshold for <UNK>
	if ft<=0:
		ft = 1
	if ft<=1:
		count =t_words["<UNK>"] - t_words["<UNK>"]*ft
		t_words["<UNK>"] -= count
		W -= count
	else:
		count =t_words["<UNK>"]*ft - t_words["<UNK>"]
		t_words["<UNK>"] += count
		W += count


	sentences = create_sentence(dev_file,t_words)

	probability_file = []
	probability_words = dict()

	for sentence in sentences:
		probability_sentence = []
		for token in sentence:
			if token not in probability_words.keys():
				probability_words[token] = t_words[token]/W
			probability_sentence.append(probability_words[token])
		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in unigram_mle:",sum(ppl)/len(ppl))
	return probability_words 

def bigram_mle(W,t_words,t_pairs,t_sentences_count, ft,dev_file):

	t_words["<BOS>"]=t_sentences_count
	t_words["<EOS>"]=t_sentences_count
	W += 2*t_sentences_count

	#setting frequency threshold for <UNK>
	if ft<=0:
		ft = 1
	if ft<=1:
		count =t_words["<UNK>"] - t_words["<UNK>"]*ft
		t_words["<UNK>"] -= count
		W -= count
		count =t_pairs[("<UNK>","<UNK>")] - t_pairs[("<UNK>","<UNK>")]*ft
		t_pairs[("<UNK>","<UNK>")] -= count
	else:
		count =t_words["<UNK>"]*ft - t_words["<UNK>"]
		t_words["<UNK>"] += count
		W += count
		count = t_pairs[("<UNK>","<UNK>")]*ft - t_pairs[("<UNK>","<UNK>")]
		t_pairs[("<UNK>","<UNK>")] += count

	

	sentences = create_sentence(dev_file,t_words)

	probability_file = []
	probability_pairs = dict()
	
	for sentence in sentences:
		sentence.insert(0,"<BOS>")
		sentence.append("<EOS>")
		probability_sentence = []
		for i in range(len(sentence)-1):
			pair = (sentence[i],sentence[i+1])

			if pair in t_pairs.keys():
				if pair not in probability_pairs.keys():
					probability_pairs[pair] = t_pairs[pair]/t_words[pair[0]]
				probability_sentence.append(probability_pairs[pair])
			else:
				if ("<UNK>","<UNK>") not in probability_pairs.keys():
					probability_pairs[("<UNK>","<UNK>")] = t_pairs[("<UNK>","<UNK>")]/t_words["<UNK>"]
				probability_sentence.append(probability_pairs[("<UNK>","<UNK>")])
		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in bigram_mle:",sum(ppl)/len(ppl))
	return probability_pairs 
	
def unigram_laplace(W,t_words,ft,dev_file,laplace):
	#setting frequency threshold for <UNK>
	if ft<=0:
		ft = 1
	if ft<=1:
		count =t_words["<UNK>"] - t_words["<UNK>"]*ft
		t_words["<UNK>"] -= count
		W -= count
	else:
		count =t_words["<UNK>"]*ft - t_words["<UNK>"]
		t_words["<UNK>"] += count
		W += count

	V = len(t_words)

	sentences = create_sentence(dev_file,t_words)

	probability_file = []
	probability_words = dict()

	for sentence in sentences:
		probability_sentence = []
		for token in sentence:
			if token not in probability_words.keys():
				probability_words[token] = (t_words[token]+laplace)/(W + (V*laplace))
			probability_sentence.append(probability_words[token])
		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in unigram_laplace:",sum(ppl)/len(ppl))
	return probability_words 

def bigram_laplace(W,t_words,t_pairs,t_sentences_count, ft,dev_file, laplace):

	t_words["<BOS>"]=t_sentences_count
	t_words["<EOS>"]=t_sentences_count
	W += 2*t_sentences_count

	#setting frequency threshold for <UNK>
	if ft<=0:
		ft = 1
	if ft<=1:
		count =t_words["<UNK>"] - t_words["<UNK>"]*ft
		t_words["<UNK>"] -= count
		W -= count
		count =t_pairs[("<UNK>","<UNK>")] - t_pairs[("<UNK>","<UNK>")]*ft
		t_pairs[("<UNK>","<UNK>")] -= count
	else:
		count =t_words["<UNK>"]*ft - t_words["<UNK>"]
		t_words["<UNK>"] += count
		W += count
		count = t_pairs[("<UNK>","<UNK>")]*ft - t_pairs[("<UNK>","<UNK>")]
		t_pairs[("<UNK>","<UNK>")] += count

	

	sentences = create_sentence(dev_file,t_words)

	probability_file = []
	probability_pairs = dict()
	vp = dict()
	
	for sentence in sentences:
		sentence.insert(0,"<BOS>")
		sentence.append("<EOS>")
		probability_sentence = []
		for i in range(len(sentence)-1):
			pair = (sentence[i],sentence[i+1])

			if pair in t_pairs.keys():
				if pair not in probability_pairs.keys():
					if pair[0] not in vp.keys():
						vp[pair[0]] = calc_vp(pair[0],t_pairs)
					probability_pairs[pair] = (t_pairs[pair] + laplace)/(t_words[pair[0]] + (vp[pair[0]]*laplace))
				probability_sentence.append(probability_pairs[pair])
			else:
				if ("<UNK>","<UNK>") not in probability_pairs.keys():
					if "<UNK>" not in vp.keys():
						vp["<UNK>"] = calc_vp("<UNK>",t_pairs)

					probability_pairs[("<UNK>","<UNK>")] = (t_pairs[("<UNK>","<UNK>")]+laplace)/(t_words["<UNK>"]+vp["<UNK>"]*laplace)
				probability_sentence.append(probability_pairs[("<UNK>","<UNK>")])
		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in bigram_laplace:",sum(ppl)/len(ppl))
	return probability_pairs
	
def calc_vp(x,pairs):
	count = 0
	for pair in pairs.keys():
		if pair[0] == x:
			count +=1
	return count


def create_sentence(dev_file,t_words):

	lines = dev_file.readlines()
	words =[]

	for line in lines:
		if line.startswith("#") or line.startswith("\n"):
			continue
		line_split = line.split("\t")
		words.append(line_split[0:2])

	del(lines)
	sentences = []
	sentence = []

	for word in words:
		if word[0] is '1':
			sentences.append(sentence)
			sentence = []
		if word[1] in t_words.keys():
			sentence.append(word[1])
		else:
			sentence.append("<UNK>")
	del(sentence)
	del(words)
	del(sentences[0])

	return sentences	

def backoff_unigram(W,t_words,dev_file,en,S1):

	sentences = create_sentence(dev_file,t_words)

	V = len(t_words)
	sum_cv = 0
	count_cv = 0
	for word, count in t_words.items():
		if count > en:
			sum_cv += count
			count_cv += 1

	if V == count_cv:
		beta = 0
	else:
		beta = ((1 - ((sum_cv - count_cv*S1)/W)) / (V-count_cv))*V


	probability_file = []
	probability_words = dict()

	for sentence in sentences:
		probability_sentence = []
		for word in sentence:
			if t_words[word] >en:
				if word not in probability_words.keys():
					probability_words[word] = (t_words[word] - S1)/W
				probability_sentence.append(probability_words[word])
			else:
				if word not in probability_words.keys():
					probability_words[word] = beta / V
				probability_sentence.append(probability_words[word])
		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in backoff_unigram:",sum(ppl)/len(ppl))
	return probability_words 


def backoff_bigram(W,t_words,t_pairs,t_sentences_count,dev_file,en,S1,S2):

	t_words["<BOS>"] = t_sentences_count
	t_words["<EOS>"] = t_sentences_count
	W += 2*t_sentences_count
	V = len(t_words)

	sum_cv = 0
	count_cv = 0
	for word, count in t_words.items():
		if count > en:
			sum_cv += count
			count_cv += 1

	if V == count_cv:
		beta = 0
	else:
		beta = ((1 - ((sum_cv - count_cv*S1)/W)) / (V-count_cv))*V

	sentences = create_sentence(dev_file, t_words)

	probability_file = []
	probability_pairs = dict()
	alpha = dict()

	for sentence in sentences:
		sentence.insert(0,"<BOS>")
		sentence.append("<EOS>")
		probability_sentence = []

		for i in range(len(sentence)-1):
			pair = (sentence[i], sentence[i+1])

			if pair in t_pairs.keys() and t_pairs[pair] > en:
				if pair not in probability_pairs.keys():
					probability_pairs[pair] = (t_pairs[pair] - S2) / t_words[pair[0]]
				probability_sentence.append(probability_pairs[pair])
			elif pair in t_pairs.keys():
				if pair not in probability_pairs.keys():
					if pair[0] not in alpha.keys():
						sum_cxv,count_cxv = calc_cxv(pair[0], t_pairs, en)
						sum_cv,count_cv,count_cv1 = calc_cv(pair[0], t_words,t_pairs,en)
						alpha[pair[0]] = (1-(sum_cxv - count_cxv*S2)/t_words[pair[0]]) / (((sum_cv-(count_cv*S1))/W)+(beta*count_cv1/V))

					if t_words[pair[1]] > en:
						probability_pairs[pair] = alpha[pair[0]] * (t_words[pair[1]]-S1)/W
					else:
						probability_pairs[pair] = alpha[pair[0]]*beta/V
				probability_sentence.append(probability_pairs[pair])
			else:
				pair = ("<UNK>","<UNK>")
				if pair in t_pairs.keys() and t_pairs[pair] > en:
					if pair not in probability_pairs.keys():
						probability_pairs[pair] = (t_pairs[pair] - S2) / t_words[pair[0]]
					probability_sentence.append(probability_pairs[pair])
				elif pair in t_pairs.keys():
					if pair not in probability_pairs.keys():
						if pair[0] not in alpha.keys():
							sum_cxv,count_cxv = calc_cxv(pair[0], t_pairs, en)
							sum_cv,count_cv,count_cv1 = calc_cv(pair[0], t_words,t_pairs,en)
							alpha[pair[0]] = (1-(sum_cxv - count_cxv*S2)/t_words[pair[0]]) / (((sum_cv-(count_cv*S1))/W)+(beta*count_cv1/V))

						if t_words[pair[1]] > en:
							probability_pairs[pair] = alpha[pair[0]] * (t_words[pair[1]]-S1)/W
						else:
							probability_pairs[pair] = alpha[pair[0]]*beta/V
					probability_sentence.append(probability_pairs[pair])

		probability_file.append(probability_sentence)

	ppl=[]
	for pf in probability_file:
		ppl.append(numpy.exp(-numpy.mean(numpy.log(pf))))
	
	print("Average of perplexity of the sentences in backoff_bigram:",sum(ppl)/len(ppl))
	return probability_pairs 

def calc_cxv(x, pairs, en):

	sum_cxv = 0
	count_cxv =0
	for pair, count in pairs.items():
		if pair[0] == x and count > en:
			sum_cxv += count
			count_cxv += 1

	return sum_cxv,count_cxv

def calc_cv(x, words, pairs, en):
	sum_cv = 0
	count_cv = 0
	count_cv1 = 0
	pairs_x = pair_startswith_x(x,pairs)
	for pair, count in pairs_x.items():
		if count <= en:
			if words[pair[1]] > en:
				sum_cv += words[pair[1]]
				count_cv += 1
			else:
				count_cv1 += 1
	return sum_cv, count_cv, count_cv1

def pair_startswith_x(x, pairs):
	pairs_x = dict()
	for pair in pairs.keys():
		if pair[0] == x:
			pairs_x[pair] = pairs[pair]

	return pairs_x

def main():
	f_name, model_name, N, path_train, path_dev, path_pickle,hp1,hp2 = argv
	N = int(N)
	hp1 = float(hp1)
	hp2 = float(hp2)
	
	try:

		training_file = open(path_train, "r", encoding="utf-8")
		dev_file = open(path_dev, "r", encoding="utf-8")
		pickle_file = open(path_pickle,"wb")

		W, t_words, t_pairs, t_sentences_count = train_data(training_file)

		#MLE
		if model_name == "mle":
			if N is 1:
				frequency_threshold = hp1
				p_dict = unigram_mle(W,t_words,frequency_threshold,dev_file)
			elif N is 2:
				frequency_threshold = hp1
				p_dict = bigram_mle(W,t_words,t_pairs,t_sentences_count, frequency_threshold,dev_file)
			else:
				print("No such N-gram in this file")
		#Laplace
		elif model_name == "laplace":
			if N is 1:
				frequency_threshold = hp1
				laplace = hp2
				p_dict = unigram_laplace(W,t_words,frequency_threshold,dev_file,laplace)
			elif N is 2:
				frequency_threshold = hp1
				laplace = hp2
				p_dict = bigram_laplace(W,t_words,t_pairs,t_sentences_count, frequency_threshold,dev_file,laplace)
			else:
				print("No such N-gram in this file")
		#Backoff model
		elif model_name == "backoff":
			if N is 1:
				en = 2
				discount_factor = hp1
				p_dict = backoff_unigram(W,t_words,dev_file,en,discount_factor)
			elif N is 2:			
				en = 2
				discount_factor1 = hp1
				discount_factor2 = hp2
				p_dict = backoff_bigram(W,t_words,t_pairs, t_sentences_count,dev_file,en,discount_factor1,discount_factor2)
			else:
				print("No such N-gram in this file")
		else:
			print("No such model in this file")
		
		dump_dict = dict()
		dump_dict["model_name"] = model_name
		dump_dict["N"] = N
		dump_dict["probability_dict"] = p_dict
		dump_dict["hyperparameter"] = (hp1,hp2)
		pickle.dump(dump_dict,pickle_file)

	finally:
		training_file.close()
		dev_file.close()
		pickle_file.close()

if __name__ == "__main__":
	main()