from bottle import route, run
from bottle import template
from bottle import get, post, request
from bottle import static_file

from nltk import word_tokenize,pos_tag,ne_chunk,data
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

import subprocess,pprint,os


ROOT="/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/NLPlay/"
HEADER = "header.html"
FOOTER = "footer.html"
CMU_ARK_DIR="/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/Tools/ark-tweet-nlp/"
ARITTER_DIR="/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/Tools/aritter-twitter-nlp/"
OPENNLP_DIR="/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/Tools/apache-opennlp-1.6.0/"
STANFORD_OPENNPL_DIR="/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/Tools/CoreNLP/lib/"

os.environ['TWITTER_NLP'] = ARITTER_DIR

def read_header_footer():
	with open(ROOT+HEADER, 'r') as header_file:
		header = header_file.read()

	with open(ROOT+FOOTER, 'r') as footer_file:
		footer = footer_file.read()

	return header,footer


@route('/')
def nlplay():
	return read_header_footer()
    # return static_file("index.html", root='/home/alexpnt/DEI/MEI/NEW_MEI/REMINDS-Internship/Experiments/NLPlay')

@get('/nlplay')
@post('/nlplay')
def nlplay():

	#get form data
	text=request.forms.get('text')
	nltk=request.forms.get('nltk')
	opennlp=request.forms.get('opennlp')
	stanfordnlp=request.forms.get('stanfordnlp')
	arittertwitternlp=request.forms.get('arittertwitternlp')
	arktwitternlp=request.forms.get('arktwitternlp')

	header,footer=read_header_footer()
	body=""

	if nltk=="on":

		tokens = word_tokenize(text)	
		pos=pos_tag(tokens)


		# Loads the serialized NEChunkParser object
		chunker = data.load('chunkers/maxent_ne_chunker/english_ace_multiclass.pickle')
		maxEnt_classifier = chunker._tagger.classifier()
		maxEnt_labels=maxEnt_classifier.labels()
		ner=ne_chunk(pos,binary=False)
		# ner.draw()

		#Stemmer
		porter_stemmer = PorterStemmer()
		porter_stems=[porter_stemmer.stem(token) for token in tokens]

		lancaster_stemmer = LancasterStemmer()
		lancaster_stems=[lancaster_stemmer.stem(token) for token in tokens]

		snowball_stemmer = SnowballStemmer("english")
		snowball_stems=[snowball_stemmer.stem(token) for token in tokens]

		#Lemmatizer		
		wordnet_lemmatizer = WordNetLemmatizer()
		wordnet_lemmas= [wordnet_lemmatizer.lemmatize(token) for token in tokens]

		#N-gram
		ngrams=nltk_word_grams(tokens)

		body+="""
		<div class="demo-separator mdl-cell--1-col"></div>
		<div class="demo-cards mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-updates mdl-card mdl-shadow--2dp mdl-cell mdl-cell--4-col mdl-cell--4-col-tablet mdl-cell--12-col-desktop">
              <div class="mdl-card__title mdl-card--expand mdl-color--teal-300">
                <h1 class="mdl-card__title-text"><b>NLTK</b></h1>
              </div>
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Tokenization:</b></h4>
		   <p>"""+pprint.pformat(tokens)+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>

		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Part-of-Speech Tagger:</b></h4>
		   <p>"""+pprint.pformat(pos)+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>

		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Named Entity Recognition:</b></h4>
		   <p>Used Corpora: ACE</p>
		   <p>Used Classifier: MaxEnt</p></br>
		   <p>Used Labels: """+pprint.pformat(maxEnt_labels)+"""</p>
		   <p>"""+pprint.pformat(ner)+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>

		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Porter Stemmer:</b></h4>
		   <p>"""+pprint.pformat(porter_stems)+"""</p>
		   <h4><b>Lancaster Stemmer:</b></h4>
		   <p>"""+pprint.pformat(lancaster_stems)+"""</p>
		   <h4><b>Snowball Stemmer:</b></h4>
		   <p>"""+pprint.pformat(snowball_stems)+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>

		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>WordNet Lemmatizer:</b></h4>
		   <p>"""+pprint.pformat(wordnet_lemmas)+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>

		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>N-grams (onegrams,bigrams,trigrams):</b></h4>
		   <p>"""+pprint.pformat(ngrams)+"""</p>
        </div>
        </div>
		
		"""

	if arktwitternlp=="on":
		proc = subprocess.Popen([CMU_ARK_DIR+"runTagger.sh",'--output-format','conll'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		result=proc.communicate()[0]

		result=result.replace("\n","</br>")
		result=result.replace("\t","&nbsp;&nbsp;&nbsp;&nbsp;")
		
		body+="""
		<div class="demo-separator mdl-cell--1-col"></div>
		<div class="demo-cards mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-updates mdl-card mdl-shadow--2dp mdl-cell mdl-cell--4-col mdl-cell--4-col-tablet mdl-cell--12-col-desktop">
              <div class="mdl-card__title mdl-card--expand mdl-color--teal-300">
		   		<h1 class="mdl-card__title-text"><b>CMU Ark Twitter NLP</b></h1>
			</div>
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Part-of-Speech Tagger:</b></h4>
		   <p>Format: conll</p>
		   <p>Tagger Model: Twitter POS</p></br>
		   <h6><b>Token POS Confidence </b></h6>
		   <p>"""+result+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>
		"""
		proc = subprocess.Popen([CMU_ARK_DIR+"runTagger.sh",'--output-format','conll','--model',CMU_ARK_DIR+'models/model.ritter_ptb_alldata_fixed.20130723'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		result=proc.communicate()[0]

		result=result.replace("\n","</br>")
		result=result.replace("\t","&nbsp;&nbsp;&nbsp;&nbsp;")
		
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Part-of-Speech Tagger:</b></h4>
		   <p>Format: conll</p>
		   <p>Tagger Model:  EMNLP 2011, Penn Treebank-style</p></br>
		   <h6><b>Token POS Confidence </b></h6>
		   <p>"""+result+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>
		"""
		proc = subprocess.Popen([CMU_ARK_DIR+"runTagger.sh",'--output-format','conll','--model',CMU_ARK_DIR+'models/model.irc.20121211'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		result=proc.communicate()[0]

		result=result.replace("\n","</br>")
		result=result.replace("\t","&nbsp;&nbsp;&nbsp;&nbsp;")
		
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Part-of-Speech Tagger:</b></h4>
		   <p>Format: conll</p>
		   <p>Tagger Model: NPSChat IRC, Penn Treebank-style</p></br>
		   <h6><b>Token POS Confidence </b></h6>
		   <p>"""+result+"""</p>
		</div>
		</div>
		"""
		proc.stdin.close()

	if arittertwitternlp=="on":
		proc = subprocess.Popen(["python2",ARITTER_DIR+"python/ner/extractEntities2.py",'--classify','--pos','--chunk'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		result=proc.communicate()[0]

		result=cutoff(result,"Average time per tweet")
		result=result.replace(" ","</br>")

		body+="""
		<div class="demo-separator mdl-cell--1-col"></div>
		<div class="demo-cards mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-updates mdl-card mdl-shadow--2dp mdl-cell mdl-cell--4-col mdl-cell--4-col-tablet mdl-cell--12-col-desktop">
              <div class="mdl-card__title mdl-card--expand mdl-color--teal-300">
		   		<h1 class="mdl-card__title-text"><b>Aritter Twitter NLP</b></h1>
			</div>
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Name Entity Recognition:</b></h4>
		   <p>Format: BIO enconding</p>
		   <p>"""+result+"""</p>
		</div>
		</div>
		"""

		proc.stdin.close()

	if opennlp=="on":

		body+="""
		<div class="demo-separator mdl-cell--1-col"></div>
		<div class="demo-cards mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-updates mdl-card mdl-shadow--2dp mdl-cell mdl-cell--4-col mdl-cell--4-col-tablet mdl-cell--12-col-desktop">
              <div class="mdl-card__title mdl-card--expand mdl-color--teal-300">
		   		<h1 class="mdl-card__title-text"><b>Apache OpenNLP</b></h1>
			</div>
		"""

		proc = subprocess.Popen([OPENNLP_DIR+"bin/opennlp",'TokenizerME',OPENNLP_DIR+'models/en-token.bin'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		tokens=proc.communicate()[0]
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Tokenization:</b></h4>
		   <p>"""+tokens+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>
		"""

		proc = subprocess.Popen([OPENNLP_DIR+"bin/opennlp",'POSTagger',OPENNLP_DIR+'models/en-pos-maxent.bin'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		pos=proc.communicate()[0]
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Part-of-Speech Tagger:</b></h4>
		   <p>"""+pos+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>
		"""

		proc = subprocess.Popen([OPENNLP_DIR+"bin/opennlp",'TokenNameFinder',OPENNLP_DIR+'models/en-ner-person.bin',OPENNLP_DIR+'models/en-ner-date.bin',OPENNLP_DIR+'models/en-ner-location.bin',OPENNLP_DIR+'models/en-ner-money.bin',OPENNLP_DIR+'models/en-ner-organization.bin',OPENNLP_DIR+'models/en-ner-percentage.bin',OPENNLP_DIR+'models/en-ner-time.bin'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		ner=proc.communicate()[0]
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Name Entity Recognition:</b></h4>
		   <p>"""+ner+"""</p>
		</div><div class="mdl-card__actions mdl-card--border"></div>
		"""

		proc = subprocess.Popen([OPENNLP_DIR+"bin/opennlp",'Parser',OPENNLP_DIR+'models/en-parser.bin',OPENNLP_DIR+'models/en-parser-chunking.bin'],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.stdin.write(text)
		syn_parsing=proc.communicate()[0]
		body+="""
		<div class="mdl-card__supporting-text mdl-color-text--grey-600">
		   <h4><b>Syntatic Parsing:</b></h4>
		   <p>"""+syn_parsing+"""</p>
		</div>
		</div>
		"""
		proc.stdin.close()

	if stanfordnlp=="on":
		with open(STANFORD_OPENNPL_DIR+"input.txt", "w") as input_file:
			input_file.write(text)

		proc = subprocess.Popen(["java","-cp",STANFORD_OPENNPL_DIR+"*","-Xmx2g","edu.stanford.nlp.pipeline.StanfordCoreNLP","-annotators","tokenize,ssplit,pos,lemma,ner,parse","-file",STANFORD_OPENNPL_DIR+"input.txt","-outputDirectory",ROOT+"static/xml"],stdout=subprocess.PIPE,stdin=subprocess.PIPE)
		proc.communicate()[0]
		proc.stdin.close()

		body+="""
		<div class="demo-separator mdl-cell--1-col"></div>
		<div class="demo-cards mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-updates mdl-card mdl-shadow--2dp mdl-cell mdl-cell--4-col mdl-cell--4-col-tablet mdl-cell--12-col-desktop">
              <div class="mdl-card__title mdl-card--expand mdl-color--teal-300">
		   		<h1 class="mdl-card__title-text"><b>Stanford CoreNLP</b></h1>
			</div>
		<div>
		    <iframe style="border:none;" width="100%" onload="this.height=screen.height;" src="input.txt.xml"></iframe> 
		</div>
		</div>
		"""

		    # <iframe style="border:none;" width="100%" onload="this.height=screen.height;" src="http://www.google.pt"></iframe>
	return header+body+footer


def nltk_word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

def cutoff(string, pattern):
		idx = string.find(pattern)
		return string[:idx if idx != -1 else len(string)]

# Static Routes
@get('/<filename:re:.*\.js>')
def javascripts(filename):
    return static_file(filename, root='static/js')

@get('/<filename:re:.*\.css>')
def stylesheets(filename):
    return static_file(filename, root='static/css')

@get('/<filename:re:.*\.(jpg|png|gif|ico)>')
def images(filename):
    return static_file(filename, root='static/images')

@get('/<filename:re:.*\.(xml|xsl)>')
def fonts(filename):
    return static_file(filename, root='static/xml')


run(host='193.136.212.4', port=8080, debug=True,reloader=True)
