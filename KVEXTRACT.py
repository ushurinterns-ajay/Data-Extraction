import pandas as pd
import os
import nltk
import numpy as np
import spacy

from tkinter import *
import tkinter.scrolledtext as st 

ftrain=os.path.join("EMAIL2QUOTE.csv")
of=os.path.join("EMAIL2QUOTEGPE.csv")
df_L=pd.read_csv(ftrain)
print(df_L.info())
r=df_L['Body'].values
Ad_mat_list = list()
total_ad_list = list()

values =dict()
fromAddr = ""
toAddr = ""
weight = ""
qnty = ""
labeled_node_list = list()
fromlist = list()
tolist = list()
weightlist = list()
qntylist = list()

def addToList(token_list,keyword,Ad_list):
	for token in token_list:
		node = list()
		node.append([token])

		from_key = [-1,-1,-1,-1]
		to_key = [-1,-1,-1,-1]
		weight_key = [-1,-1,-1,-1]
		qnty_key = [-1,-1,-1,-1]

		if keyword == 'from':
			from_key = [1,1,1,1]
		elif keyword == 'to':
			to_key = [1,1,1,1]
		elif keyword == 'weight':
			weight_key = [1,1,1,1]
		elif keyword == 'qnty':
			qnty_key = [1,1,1,1]

		edges = list()
		edges.append(from_key)
		edges.append(to_key)
		edges.append(weight_key)
		edges.append(qnty_key)
		node.append(edges)
		Ad_list.append(node)
	return Ad_list

def findFromAdd():
	for sentence in labeled_node_list:
		for ind,token in enumerate(sentence):
			# print(token)
			word = token.split('-')[1]
			cur_word = re.sub('[\W_]+', '', word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if cur_stem in ('from','origin','collect','pickup'):
				position = eval(token.split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				return sen_pos,word_pos
			elif(cur_stem=='pick' and next_stem=='up'):
				position = eval(sentence[ind+1].split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				return sen_pos,word_pos
	return -1,-1
def findToAdd():
	for sentence in labeled_node_list:
		for ind,token in enumerate(sentence):
			# print(token)
			word = token.split('-')[1]
			cur_word = re.sub('[\W_]+', '', word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if cur_stem in ('to','deliveri','deiv','destin'):
				position = eval(token.split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				return sen_pos,word_pos
			elif(cur_stem=='ship' and next_stem=='to'):
				position = eval(sentence[ind+1].split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				return sen_pos,word_pos
	return -1,-1


def findQnty():
	for sentence in labeled_node_list:
		for ind,token in enumerate(sentence):
			# print(token)
			word = token.split('-')[1]
			cur_word = re.sub('[\W_]+', '', word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if cur_stem in ('pc','piec','pallet','ctn','plt','carton','palett'):
				position = eval(token.split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				if word_pos != 1:
					word_pos -= 1
				return sen_pos,word_pos
	return -1,-1

def findWeight():
	for sentence in labeled_node_list:
		for ind,token in enumerate(sentence):
			# print(token)
			word = token.split('-')[1]
			cur_word = re.sub('[\W_]+', '', word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if cur_stem == 'weight':
				position = eval(token.split('-')[0])
				sen_pos = position[0]
				word_pos = position[1]
				return sen_pos,word_pos
	return -1,-1

def insertFrom(newFromAddress,sen_pos,word_pos,flag):
	if(sen_pos == -1):
		if flag == 0:
			newFromAddress = 'from,address,'+ newFromAddress
			newfrom = newFromAddress.split(',')
		elif flag == 1:
			newFromAddress = 'to,address'+ newFromAddress
			newfrom = newFromAddress.split(',')
		elif flag == 2:
			newFromAddress = 'weight '+ newFromAddress
			newfrom = newFromAddress.split(' ')
		elif flag ==3:
			newFromAddress = 'quantity '+ newFromAddress
			newfrom = newFromAddress.split(' ')
		cur_list = list()
		sen_pos = len(labeled_node_list)
		for ind,token in enumerate(newfrom):
			ner='O'
			val=str(token).replace('-', '').replace("'", '').replace('(', '').replace(')', '')
			if SNERTAGGER(token):
				ner=SNERTAGGER(token)[0]
			pos=POSTAGGER(token)[0]
			cur_list.append('(' + str(sen_pos+1) + ',' +str(ind+1) + ')' + '-' + val + '-' + pos + '-' + ner)
		labeled_node_list.append(cur_list)
		return cur_list
	else:
		result = list()
		if flag == 2 or flag == 3:
			newfrom = newFromAddress.split(' ')
		else:
			newfrom = newFromAddress.split(',')
		spam_lenth = len(newfrom)
		# print("spam_lenth:",spam_lenth)
		# print(len(labeled_node_list[sen_pos-1])+spam_lenth)
		index = 0
		for i in range (len(labeled_node_list[sen_pos-1])+spam_lenth):
			if i < word_pos:
				continue
			elif index < spam_lenth:
				token = newfrom[index]
				ner='0'
				val=str(token).replace('-', '').replace("'", '').replace('(', '').replace(')', '')
				if SNERTAGGER(token):
					ner=SNERTAGGER(token)[0]
				pos=POSTAGGER(token)[0]
				tok = '(' + str(sen_pos) + ',' +str(i+1) + ')' + '-' + val + '-' + pos + '-' + ner
				result.append(tok)
				labeled_node_list[sen_pos-1].insert(i,tok)
				# print(tok,i)
				index = index + 1
			else:
				# print(i)
				tokens = labeled_node_list[sen_pos-1][i].split('-')
				tokens[0]='('+str(sen_pos) + ',' +str(i+1) + ')'
				seperator='-'
				token = seperator.join(tokens)
				labeled_node_list[sen_pos-1][i] = token
		# print("New:-",labeled_node_list)
		return result

	


def shiftByk(keyword_list,k):
	new_list = list()
	for token in keyword_list:
		tokens = token.split('-')
		position = eval(tokens[0])
		tokens[0]='('+str(position[0]) + ',' +str(position[1]+k) + ')'
		seperator = '-'
		token = seperator.join(tokens)
		new_list.append(token)
	return new_list



class MyWindow:
    def __init__(self, win):
        self.lbl1=Label(win, text='From : ')
        self.lbl2=Label(win, text='To :')
        self.lbl3=Label(win, text='Weight:')
        self.lbl4=Label(win, text='Dimensions:')
        self.lbl5=Label(win, text='Qnty:')
        self.lbl6=Label(win, text='Email:')
       
        self.t1=Entry(bd=3)
        self.t2=Entry()
        self.t3=Entry()
        self.t4=Entry()
        self.t5=Entry()
        self.lbl1.place(x=100, y=30)
        self.t1.place(x=200, y=30)
        self.lbl2.place(x=100, y=60)
        self.t2.place(x=200, y=60)
        self.lbl3.place(x=100, y=90)
        self.t3.place(x=200, y=90)
        self.lbl4.place(x=100, y=120)
        self.t4.place(x=200, y=120)
        self.lbl5.place(x=100, y=150)
        self.t5.place(x=200, y=150)
        self.lbl6.place(x=450,y=10)
        # self.lbl7.place(x=200,y=250)
        self.b1=Button(win, text='Correct', command=self.writeToLog)
        self.b1.place(x=200, y=200)
        self.b2=Button(win, text='extract', command=self.FillForm)
        self.b2.place(x=100, y=200)
        self.fromlist = list()
        self.tolist = list()
        self.weightlist = list()
        self.qntylist = list()

    def FillForm(self):
    	self.t1.delete(0, END)
    	self.t1.insert(END,str(values['from']))
    	self.t2.delete(0, END)
    	self.t2.insert(END,str(values['to']))
    	self.t3.delete(0, END)
    	self.t3.insert(END,str(values['weight']))
    	self.t4.delete(0, END)
    	self.t4.insert(END,str(values['dimension']))
    	self.t5.delete(0, END)
    	self.t5.insert(END,str(values['qnty']))
        # self.v = r[count]
        # self.lbl7.config(text=str(r[count]))
        
    def writeToLog(self):
        print(self.t1.get())
        print(self.t2.get())
        print(self.t3.get())
        print(self.t4.get())
        print(self.t5.get())
        f_sen,f_word=-1,-1
        t_sen,t_word=-1,-1
        w_sen,w_word=-1,-1
        q_sen,q_word=-1,-1
        # fromlist = list()
        # tolist = list()
        # weightlist = list()
        # qntylist = list()
        # print(fromlist)
        # print(tolist)
        # print(qntylist)
        # print(weightlist)
        if fromAddr != self.t1.get():
	        f_sen,f_word = findFromAdd()
	        print("From position : ",f_sen,f_word)
	        self.fromlist = insertFrom(self.t1.get(),f_sen,f_word,0)
	        print("new From : ",self.fromlist)

        if toAddr != self.t2.get():
        	t_sen,t_word = findToAdd()
        	print("To position : ",t_sen,t_word)
        	if f_sen == t_sen and f_word > t_word and f_sen != -1:
        		self.fromlist = shiftByk(self.fromlist,len(self.t2.get().split(',')))
        	self.tolist = insertFrom(self.t2.get(),t_sen,t_word,1)
        	print("New To : ",self.tolist)

        if weight != self.t3.get():
        	w_sen,w_word = findWeight()
        	print("Weight position : ",w_sen,w_word)
        	if f_sen == w_sen and f_word > w_word and f_sen != -1:
        		self.fromlist = shiftByk(self.fromlist,len(self.t3.get().split(' ')))
        	if t_sen == w_sen and t_word > w_word and t_sen != -1:
        		self.tolist = shiftByk(self.tolist,len(self.t3.get().split(' ')))
        	self.weightlist = insertFrom(self.t3.get(),w_sen,w_word,2)
        	print("new Weight : ",self.weightlist)

        if qnty != self.t5.get():
        	q_sen,q_word = findQnty()
        	print("Qnty position : ",q_sen,q_word)
        	if f_sen == q_sen and f_word > q_word and f_sen != -1:
        		self.fromlist = shiftByk(self.fromlist,len(self.t5.get().split(' ')))
        	if t_sen == q_sen and t_word > q_word and t_sen != -1:
        		self.tolist = shiftByk(self.tolist,len(self.t5.get().split(' ')))
        	if w_sen == q_sen and w_word > q_word and w_sen != -1:
        		self.weightlist = shiftByk(self.weightlist,len(self.t5.get().split(' ')))

        	self.qntylist = insertFrom(self.t5.get(),q_sen,q_word,3)
        	print("New Qnrt : ",self.qntylist)
        # print(fromlist)
        # print(tolist)
        # print(qntylist)
        # print(weightlist)
        # print("adjm list")
        # Ad_list = addToList(fromlist,"from",Ad_list)
        # Ad_list = addToList(tolist,"to",Ad_list)
        # Ad_list = addToList(weightlist,"weight",Ad_list)
        # Ad_list = addToList(qntylist,"qnty",Ad_list)
        # print(Ad_list)


        close()

def close():
    window.destroy()
    
    
    
    






def safely_execute(func):
    def func_wrapper(*args, **kwargs):
        try:
           return func(*args, **kwargs)
        except Exception as e:
            print(e) # or write to log
            return None
    return func_wrapper
@safely_execute
def get_x_info(model):
    print("hi")
'''
j=np.array([[0,1],[-1,0]])
#print(j.shape)
m = np.dot(j, [[0,1,2,0,1,2,0,1,2],[0,0,0,1,1,1,2,2,2]])
#print(m)


for  x in range(3,1,-1):
    print("hi",x)
'''
def MIlaplacian(adjm):
    #laplacian: diagonal entries are all 1
    #connected nodes have 1/degree else 0
    degree_list=adjm.sum(axis=1).astype(np.float32)
    #sum up degree for each node, possibility of disconnected node
    degree_list=np.where(degree_list<=0,-1,degree_list)
    coeff=-np.where(degree_list<0,0,1/degree_list)
    for row in range(adjm.shape[0]):
        adjm[row,:]*=coeff[row]
    np.fill_diagonal(adjm,1)
    return adjm
nlp = spacy.load('en_core_web_sm')
#def NLTKNERTAGGER(txt):
    #return [x[2] for x in tree2conlltags(ne_chunk(nltk.pos_tag(word_tokenize(txt)))) if x[2] != 'O']
def SNERTAGGER(txt):
    doc=nlp(txt)
    chunks=[token.label_ for token in doc.ents]
    return chunks
def POSTAGGER(txt):
    doc=nlp(txt)
    chunks=[token.pos_ for token in doc]
    #print(nltk.pos_tag(nltk.word_tokenize(txt)))
    #print(spost.tag(nltk.word_tokenize(txt)))
    return chunks


# print(r[0].split())

def sweep_email(index,text):
	count = 0
	for ic, value in enumerate(text.split()):
		ner='O'
		if value in [':']:
			continue
		val=str(value).replace('-', '').replace("'", '').replace('(', '').replace(')', '')
		if SNERTAGGER(value):
			ner=SNERTAGGER(value)[0]
		#else:
		pos=POSTAGGER(value)[0]
		count = count + 1
		yield '(' + str(index) + ',' +str(count) + ')' + '-' + val + '-' + pos + '-' + ner
desired_tokens=['PROPN','ADP','NUM']
text=r[0]

def MI(index, colmax,node, nodelist):
    if index==0:
        return None
    if nodelist[index].split('-')[2] == 'NUM':
        return None
    if nodelist[index].split('-')[3]=='GPE' and nodelist[index-1].split('-')[2]=='ADP':
        return nodelist[index-1], node
    else:
        if nodelist[index].split('-')[2] == 'PROPN' and nodelist[index - 1].split('-')[2] == 'ADP':
            return nodelist[index-1],node
        else:
             return None

def det_label(node,index, desired_node_list):
    if index!=0:
        if node.split('-')[2]=='NUM' and desired_node_list[index-1].split('-')[3]=="GPE":
            nn=node.split('-')[0] + '-' + node.split('-')[1] + '-' + 'ZIPCODE' + '-' + node.split('-')[3]
            # print("zip found" ,nn)
            return nn
        else:
            return  node
    else:
        return node


nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer# 
import re

porter_stemmer = PorterStemmer()

def getQnty(node_list):
	node = list()
	for sentence in node_list:
		for ind,token in enumerate(sentence):
			cur_word = token.split('-')[1].lower()
			cur_word = re.sub('[\W_]+', '', cur_word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			pre_stem = ""
			if ind != 0 :
				pre_stem = sentence[ind-1].split('-')[1].lower()
				pre_stem = re.sub('[\W_]+', '', pre_stem)
				pre_stem = porter_stemmer.stem(pre_stem)
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if len(sentence) == ind-1 or ind+1 == len(sentence):
				continue
			elif (token.split('-')[3] =='CARDINAL') and next_stem in ('pc','piec','pallet','ctn','plt','carton','palett'):
				node.clear()
				node.append(token)
				node.append(sentence[ind+1])
				return node
			elif (cur_stem in ('pc','piec','pallet','ctn','plt','carton','palett') and sentence[ind+1].split('-')[3] =='CARDINAL'):
				node.clear()
				node.append(sentence[ind+1])
				node.append(token)
				return node
			elif [True for x in ('pc','piec','pallet','ctn','plt','carton','palett') if x in cur_stem] and re.search('\d', cur_stem):
				node.clear()
				node.append(token)
				return node
			elif cur_stem in ('pc','piec','pallet','ctn','plt','carton','palett'):
				node.clear()
				node.append(token)

				# return node
	return node

def getWeight(nodelist):
	result = list()
	keys = ('kg','lb','pound')
	for sentence in nodelist:
		for ind,token in enumerate(sentence):
			cur_stem = porter_stemmer.stem(token.split('-')[1].lower())
			next_stem = ""
			if ind+1 < len(sentence):
				next_stem = porter_stemmer.stem(sentence[ind+1].split('-')[1].lower())
			if 'kg' in cur_stem or 'lb' in cur_stem or 'pound' in cur_stem:
				result.append(token)
			elif (token.split('-')[3] =='CARDINAL' or token.split('-')[2] == 'NUM') and next_stem in keys:
				result.append(token)
	return result

def getToAdd(nodelist):
	result = list()
	ans = 0
	for sentence in nodelist:
		for ind,token in enumerate(sentence):
			if '_' in token.split('-')[1]:
				continue
			if ans == 1 and (token.split('-')[1] == '.' or token.split('-')[1].lower() in('from','to:')):
				return result
			elif ans == 1:
				result.append(token)
				if(token.split('-')[1].endswith(".")) or len(result) >= 5:
					return result
			if token.split('-')[1].lower() in ('to:','to','delivery','destination','destination:') and sentence[ind+1].split('-')[2] not in ('AUX','INTJ'):
				ans = 1
	return result

def getFromAdd(nodelist):
	result = list()
	ans = 0
	for sentence in nodelist:
		for ind,token in enumerate(sentence):
			word = token.split('-')[1]
			cur_word = re.sub('[\W_]+', '', word)
			cur_stem = porter_stemmer.stem(cur_word)
			next_stem = ""
			pre_stem = ""
			if ind != 0 :
				pre_stem = sentence[ind-1].split('-')[1].lower()
				pre_stem = re.sub('[\W_]+', '', pre_stem)
				pre_stem = porter_stemmer.stem(pre_stem)
			if ind+1  < len(sentence):  
				next_stem = sentence[ind+1].split('-')[1].lower()
				next_stem = re.sub('[\W_]+', '', next_stem)
				next_stem = porter_stemmer.stem(next_stem)
			if '_' in word or token.split('-')[2] in ('PUNCT') or cur_word in('up','in'):
				continue
			if ans == 1 and (cur_stem in ('to','deliveri','destin','.','deliv')):
				if(len(result) == 0):
					ans = 0
				else:
					return result
			elif ans == 1:
				if(token.split('-')[2] not in ('CCONJ')):
					result.append(token)
				if word.endswith('.') or len(result) >= 5:
					return result
			elif (cur_stem in ('from','origin','collect','pickup','door') and next_stem not in('door','to') and pre_stem != 'to') or (cur_stem == 'pick' and next_stem =='up' and pre_stem != 'not'):
				ans = 1
	return result

count  = 0

for text in r:
    sentances = sent_tokenize(text)

    # print()
    # print(sentances)
    # print()
    # # exit()
    labeled_node_list = list()
    for index, sentence in enumerate(sentances):
        labeled_node_list.append([ x for x in list(sweep_email(index+1,sentence))])
    # print("old:-",labeled_node_list)
    # print("QTY: ",count,getQnty(labeled_node_list))
    fromlist = getFromAdd(labeled_node_list)
    tolist = getToAdd(labeled_node_list)
    weightlist = getWeight(labeled_node_list)
    qntylist = getQnty(labeled_node_list)
    # print('count',count,':',qntylist)
    seperator = ','
    if(len(fromlist) > 0):
    	fromAddr = seperator.join(x.split('-')[1] for x in fromlist)
    else: 
    	fromAddr = "NULL"
    if len(tolist) > 0:
    	toAddr = seperator.join(x.split('-')[1] for x in tolist)
    else:
    	toAddr = "NULL"
    seperator = '-'
    if len(weightlist) > 0:
    	weight =seperator.join(x.split('-')[1] for x in weightlist)
    else:
    	weight = "NULL"
    if len(qntylist) > 0:
    	qnty = seperator.join(x.split('-')[1] for x in qntylist)
    else:
    	qnty = "NULL"

    #extracted values
    values['from'] = fromAddr
    values['to'] = toAddr
    values['qnty'] = qnty
    values['dimension'] = "NULL"
    values['weight'] = weight

    # #GUI creation

    window=Tk()
    mywin=MyWindow(window)
    window.title('Data Extration')
    window.geometry("1300x800+10+10")
    # mail = Message(window,text=text)
    # mail.config(anchor='nw',width='600',font=('times', 12, 'italic'))
    # mail.pack()
    text_area = st.ScrolledText(window,width = 80,height=20,font=('times', 12, 'italic'))
    text_area.grid(column=10,pady=10,padx=500)
    text_area.insert(INSERT,text)
    text_area.configure(state ='disabled') 
    window.mainloop()

    print("Exit from GUI")
    Ad_list = list()
    if len(mywin.fromlist) == 0:	
    	Ad_list = addToList(fromlist,"from",Ad_list)
    else:
    	Ad_list = addToList(mywin.fromlist,"from",Ad_list)
    if len(mywin.tolist) == 0:
    	Ad_list = addToList(tolist,"to",Ad_list)
    else:
    	Ad_list = addToList(mywin.tolist,"to",Ad_list)
    if len(mywin.weightlist) == 0:
    	Ad_list = addToList(weightlist,"weight",Ad_list)
    else:
    	Ad_list = addToList(mywin.weightlist,"weight",Ad_list)
    if len(mywin.qntylist) == 0:
    	Ad_list = addToList(qntylist,"qnty",Ad_list)
    else:
    	Ad_list = addToList(mywin.qntylist,"qnty",Ad_list)
    print("Adjcent LIst:- ",Ad_list)
    adj_matrix=np.zeros((len(Ad_list),4))
    for row, column in enumerate(Ad_list):
    	# print(column[1])
    	for edgeno, edge in enumerate(column[1]):
    		# print(edge)
    		if edge[0]!=-1:
    			adj_matrix[row,edgeno]=1
    print("Adjcent Matrix:- ",adj_matrix)
    count =count + 1
    Ad_mat_list.append(adj_matrix)
    total_ad_list.append(Ad_list)
exit()

