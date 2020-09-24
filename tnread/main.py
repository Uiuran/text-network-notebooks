# -*- coding: utf-8 -*-

import os
from pathlib import Path
import networkx as nx
from fa2 import ForceAtlas2
import igraph as ig  # TODO: replace networkx with igraph if all works well
from collections import Counter
import louvain  
import community
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import datetime
from textblob import TextBlob
from nltk.stem.snowball import SnowballStemmer
from html.parser import HTMLParser
from nltk.corpus import stopwords as stopw
import re
from nltk.stem import PorterStemmer
from nltk.stem.rslp import RSLPStemmer
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing

from .utils import *
from .discourse import *

class MLStripper(HTMLParser):
    """Utility class for stripping HTMTL from web texts.
    """

    def error(self, message):
        pass

    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
         return ''.join(self.fed)


def strip_tags(self):
    """Strip HTML tags from text."""
    s = MLStripper()
    s.feed(self.text)

    return s.get_data()


class TextNetwork:
    """
    Full creation process of Discourse Bias Text Network Analysis
    ------------------------------------------------------------

    A Python implementation of the research methodology for measuring discourse bias presented by Dmitry Paranyushkin
    in https://towardsdatascience.com/measuring-discourse-bias-using-text-network-analysis-9f251be5f6f3.

    The technical details are described in the 2011 whitepaper Identifying the Pathways for Meaning Circulation using
    Text Network Analysis: https://noduslabs.com/publications/Pathways-Meaning-Text-Network-Analysis.pdf

    Intermediary results from steps of the process is documented in a returned Pandas DataFrame object together with 
    experiment settings. 
    
    The produced text network graph is saved as aGEXF-format file per default in a default folder named 'gexf/' in the 
    current working directory.

    Example usage:
    --------------

        import discousebias as dib
        text = '''In this work we propose a method and algorithm for identifying the pathways for meaning
                  circulation within a text. This is done by visualizing normalized textual data as a graph and
                  deriving the key metrics for the concepts and for the text as a whole using network
                  analysis. '''

        experiment = dib.TextNetwork(text, name="pathways_of_meaning")
        data = experiment.run()

        # Inspect pre-processed text
        print(experiment.no_stops_paras) 
        
        # Inspect settings and intermediary results
        print(data) 


    Settings (default):
    -------------------
        lang="en", defines what default stopwords list and stemmer to use.

        window=4, defines the word-window used when setting weights for the text network

        save_gexf=True, wether or not to save the created graph to local folder

        gefx_path="gexf/", folder in which to save graph file. Creates it if not present.

        plot=True, prints a matplotlib based chart of topn_nodes distribution in topn_comms.
            Suitable for when using Jupyter Notebooks. Note: put '%matplotlib inline' at the
            top of the notebook to ensure that the chart is actually plotted to output.

        topn_nodes=4, the number of nodes with the highest betweenness centrality ecore to use for computation of
            graph structure.

        topn_comms=4, the number of communities to use for calculation of graph structure.

        strip_html=False, set to true if you want to strip the input text from HTML tags before processing.

        max_nodes=150, cut-off for number of nodes to be used for graph creation. Infranodus.com number is 150.

        stemmer="porter", choose between hunspell or porter stemmer. Pyhunspell better, but complex installation.

        connect_paragraphs=True, optional, creates a more connected graph, see Paranyushkin, 2011

        emphasize_plot_turns=False

            By default the algorithm calculates betweenness centrality for an undirected graph, 
            which works better for identifying the most important keywords in the whole narrative.

            However, if you set emphasize_plot_turns=True, then betweenness centraliy will be calculated 
            for the directed graph, following the text's direction. This will emphasize the turning points 
            in the plot of the narrative, helping you identify interesting keywords, which are may be not 
            so frequent, but important to the flow of the narrative. This may be a more interesting setting 
            for topic modelling when you want to emphasize parts of the text that are not so evident from 
            the first sight or using other text analysis tools. 

            It follows from here that bias detection with this setting as True will make all your texts 
            a bit "less" biased and more "diverse", because you're checking dispersal of the kind of key terms 
            that are a bit less crucial to the meaning of the text as a totality. 

            It also makes sense to leave this setting as False for languages with a very specific word order 
            like German, as the algorithm considers the standard direction of reading from left to right, 
            so in case of German, setting this to True would link all your verbs (in the end of a sentence) 
            to the next sentences.  

    """

    def __init__(self, text, textname="Unnamed", lang="en", window=4, save_gexf=True,
                 gexf_path="gexf/", plot=True, topn_nodes=4, topn_comms=4, strip_html=False,
                 max_nodes=150, stem=True, stemmer="porter", connect_paragraphs=False,
                 emphasize_plot_turns=False, verbose=False):
        """Initiate the process using settings from the passed in parameters.

        :param text: string
        :param textname: identifier of text, e.g. filename, URL, headline
        :param lang: ISO 639-1 two-letter language code, e.g. "sv" or "en"
        """
        self.verbose = verbose
        self.stem=stem
        self.stats = {}
        self.text = text
        self.textname = textname

        # settings
        self.lang = lang
        self.window = window
        self.save_gexf = save_gexf
        self.gexf_path = gexf_path
        self.plot = plot
        self.topn_nodes = topn_nodes
        self.topn_comms = topn_comms
        self.strip_html = strip_html
        self.max_nodes = max_nodes
        self.stemmer = stemmer
        self.connect_paragraphs = connect_paragraphs
        self.bc_directed = emphasize_plot_turns # Calculate betweenness centrality on a directed graph?
        
        # Methodological outputs
        self.stopwords = None

        self.paragraphs = None
        self.paragraphs_cleaned = None
        self.lowerc_paras = None
        self.no_stops_paras = None
        self.stemmed_paras = None

        self.origWords = None
        self.origUniqueW = None
        self.origUniqeLcW = None
        self.stemmedUniqLcW = None

        self.nodes_df = None
        self.edges_df = None

        self.partitions = None  # in cutoffGraph

        self.origGraph = None
        self.cutoffGraph = None
        self.igraph = None

        self.entropy = None

        self.E = None  # Entropy
        self.M = None  # Modularity
        self.G = None  # Percentage of nodes G in the giant component of the graph
        self.C = None  # Percentage of nodes C in the community with the most nodes
        self.biasIndex = None
        self.df = None  # Pandas DataFrame

    def __str__(self):
        pass

    def run(self):
        """Run the whole research process on text submitted to class instance.

        :return: Pandas DataFrame with results and statistics.
        """
        if self.strip_html:
            self.text = strip_html()

        self.load_stopwords()
        self.split_paragraphs()
        self.remove_numbers_and_special_characters()
        self.lowercase_paragraphs()
        self.remove_stopwords()
        self.stem_words()
        self.text_network()
        self.cutoff_graph_max_nodes()
        self.communities_and_betweenness_centrality()
        self.discourse_diversity()

        if self.save_gexf:
            self.write_graph_to_gexf_file()
        data = self.generate_stats_dataframe()

        return data

    def load_stopwords(self):
        """Load stopwords from file and transform into list.

        Currently only Swedish stopwords implemented.

        lang:sv
            List of stopwords from https://github.com/stopwords-iso/stopwords-sv/blob/master/stopwords-sv.txt
            and also added "http" to the list.


        return: list
        """
        self.stats["lang"] = self.lang  # Keep track of this setting
        my_path = os.path.dirname(__file__)
        if self.lang == "sv":
            filepath = os.path.join(my_path, "stopwords/stopwords-sv.txt")
            stopwords = [w.strip() for w in open(filepath).readlines()]
            stopwords.extend(["http"])
            self.stats["nr_stopw"] = len(stopwords)
            self.stats["stopwords_filew"] = filepath.rpartition("/")[2]

            self.stopwords = stopwords
            return self.stopwords

        elif self.lang == "en":
            filepath = os.path.join(my_path, "stopwords/stopwords-en-en.txt")
            stopwords = [w.strip() for w in open(filepath).readlines()]
            self.stats["nr_stopw"] = len(stopwords)

            self.stopwords = stopwords
            return self.stopwords
        elif self.lang == "pt":
            filepath = os.path.join(my_path, "stopwords/stopwords-pt.txt")
            stopwords = [w.strip() for w in open(filepath).readlines()]
            self.stats["nr_stopw"] = len(stopwords)
            self.stopwords = stopwords
            return self.stopwords
            
        else:
            print("Only Swedish and English stopwords implemented yet.")  # TODO: Warn logging?

    def split_paragraphs(self):
        """Splits paragraphs, remove puctuation and return list of paragraps split into list of words.

        """
        self.paragraphs = re.split(r"\n\n|\r\n\r\n|\n", self.text)

        tokenized_paragraphs = []
        for para in self.paragraphs:
            tokenized_sentences = []
            blob = TextBlob(para)
            for s in blob.sentences:
                tokenized_sentences.extend(s.words)
            tokenized_paragraphs.append(tokenized_sentences)

        self.paragraphs = tokenized_paragraphs

    def remove_numbers_and_special_characters(self):
        """Normalizes text and returns list of lists with words."""

        new_list_of_lists = []
        for para in self.paragraphs:
            clean_para = []
            for t in para:
                # Remove numbers e.g. 4.23, 1926, I. and XV. (roman numerals common in academic chaptering)
                t = re.sub(r'\b\d+(?:\.\d+)?\s+|\d+|[IVXLCDM]+\.', '', t)

                # remove special chars with possible encoding problems
                t = re.sub(u"[\u2018\u2019\u2013\u201c\u201d\u200e]", "", t)
                clean_para.append(t)

            clean_para2 = []
            for t in clean_para:
                if t == "":  # Python RE module returns empty strings when re.sub is matching
                    continue
                else:
                    clean_para2.append(t)

            new_list_of_lists.append(clean_para2)

            self.paragraphs_cleaned = new_list_of_lists

    def lowercase_paragraphs(self):
        """Take list of lists of paragraphs and return them with all words lowercase.
        """
        all_new_paras = []
        for para in self.paragraphs_cleaned:
            new_para = []
            for token in para:
                new_para.append(token.lower())
            all_new_paras.append(new_para)

        self.lowerc_paras = all_new_paras

    def remove_stopwords(self):
        """Removes stopwords from words contained in list of lists of paragraphs.
        """
        all_new_paras = []
        for para in self.lowerc_paras:
            new_para = []
            for t in para:
                if t.lower() in self.stopwords:
                    continue
                else:
                    new_para.append(t)
            if len(new_para) > 0:
                all_new_paras.append(new_para)

        self.no_stops_paras = all_new_paras
        self.stats["no_stops_paras"] = self.no_stops_paras

    def stem_words(self):
        """Stem words contained in list of lists of paragraphs.
        """

        all_new_paras = []

        if self.stem:

            if self.lang == "sv" and self.stemmer == "pyhunspell":
                import hunspell  # Will probably not work for anyone on mac os x. see setup_hunspell_mac_osx.md
                usr_dir = os.path.expanduser("~")
                sv_dic = Path(usr_dir + '/Library/Spelling/sv_SE.dic')
                if sv_dic.is_file():  # assuming .aff file is also present
                    hobj = hunspell.HunSpell(usr_dir + '/Library/Spelling/sv_SE.dic',
                                             usr_dir + '/Library/Spelling/sv_SE.aff')
                    for para in self.no_stops_paras:
                        new_para = []
                        for token in para:
                            bytes_token_list = hobj.stem(token)
                            if len(bytes_token_list) < 1:
                                continue
                            else:
                                string_token = bytes_token_list[0].decode("utf-8")
                                new_para.append(string_token)
                        if len(new_para) > 0:
                            all_new_paras.append(new_para)

                else:
                    print("Check that file sv_SE.dic and sv_SE.aff exists in ~/Library/Spelling/")


            elif self.lang == "sv" and self.stemmer == "porter":
                stemmer = SnowballStemmer("swedish")
                for para in self.no_stops_paras:
                    new_para = [stemmer.stem(token) for token in para]
                    all_new_paras.append(new_para)

            elif self.lang == "en" and self.stemmer == "pyhunspell":
                import hunspell  # Will probably not work for anyone on mac os x. see setup_hunspell_mac_osx.md
                usr_dir = os.path.expanduser("~")
                en_dic = Path(usr_dir + '/Library/Spelling/en_US.dic')
                if en_dic.is_file():  # assuming .aff file is also present
                    hobj = hunspell.HunSpell(usr_dir + '/Library/Spelling/en_US.dic',
                                             usr_dir + '/Library/Spelling/en_US.aff')
                    for para in self.no_stops_paras:
                        new_para = []
                        for token in para:
                            bytes_token_list = hobj.stem(token.encode("utf-8"))
                            if len(bytes_token_list) < 1:
                                continue
                            else:
                                string_token = bytes_token_list[0].decode("utf-8")
                                new_para.append(string_token)
                        if len(new_para) > 0:
                            all_new_paras.append(new_para)

                else:
                    print("Check that file en_US.dic and en_US.aff exists in ~/Library/Spelling/")

            elif self.lang == "en" and self.stemmer == "porter":
                stemmer = PorterStemmer()
                
                for para in self.no_stops_paras:
                    new_para = []
                    for token in para:
                        tokenstem = stemmer.stem(token.lower())
                        if tokenstem in self.stopwords:
                            continue
                        else:
                            new_para.append(tokenstem)
            
                    if len(new_para) > 0:
                        all_new_paras.append(new_para)
            elif self.lang == "pt":
                stemmer = RSLPStemmer()
                
                for para in self.no_stops_paras:
                    new_para = []
                    for token in para:
                        tokenstem = stemmer.stem(token.lower())
                        if tokenstem in self.stopwords:
                            continue
                        else:
                            new_para.append(tokenstem)
            
                    if len(new_para) > 0:
                        all_new_paras.append(new_para)
        else:
            for para in self.no_stops_paras:
                new_para=[]
                for token in para:
                    token = token.lower()
                    if token in self.stopwords:
                        continue
                    else:
                        new_para.append(token)
                if len(new_para) > 0:
                    all_new_paras.append(new_para)
        
        self.stemmed_paras = all_new_paras
        self.stats["stemmed_paras"] = self.stemmed_paras

        # word statistics at this point
        words = []
        for para in self.stemmed_paras:
            for w in para:
                words.append(w)
        self.stemmedUniqLcW = len(words)
        self.stats["stemmedUniqLcW"] = self.stemmedUniqLcW

    def text_statistics(self):
        """Compute basic statistics about the text in different stages of the process."""

        words = []
        for para in self.paragraphs_cleaned:
            for word in sent:
                words.append(word)

        self.origWords = words
        self.origUniqueW = list(set(words))
        self.origUniqeLcW = list(set([w.lower() for w in words]))
        self.stats["origWords"] = len(self.origWords)
        self.stats["origUniqueW"] = len(self.origUniqueW)
        self.stats["origUniqueLcW"] = len(self.origUniqeLcW)

    def text_network(self):
        """Create the text network as networkx graph object from list of lists of sentences.
        """
        G = nx.DiGraph()
        GUn = nx.Graph()

        now = datetime.datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M')
        self.stats["created_at"] = ts

        G.name = self.textname + "_" + now.strftime('%Y-%m-%d')

        self.stats["window"] = self.window

        # First run, create all nodes
        # this is if we want to split them by sentences
        # for sent in self.stemmed_sents:
        for paragraph in self.stemmed_paras:
            for word in paragraph:
                if word == "'s" or word == "n't":
                    paragraph.remove(word)

        for paragraph in self.stemmed_paras:
            for word in paragraph:
                if word not in G.nodes():
                    G.add_node(word)
                if word not in GUn.nodes():
                    GUn.add_node(word)


        for para in self.stemmed_paras:
            fourwords = [para[i:i + self.window] for i in range(len(para))]
            for seq in fourwords:
                distance_weight = self.window -1
                if len(seq) != 1:
                    w1 = seq[0]

                    for w2 in seq[1:]:
                        if w1 != w2:
                            if G.has_edge(w1, w2):
                                G[w1][w2]['weight'] += distance_weight
                            else:
                                G.add_edge(w1, w2, weight=distance_weight)

                            if GUn.has_edge(w1, w2) or GUn.has_edge(w2, w1):
                                GUn[w1][w2]['weight'] += distance_weight
                            else:
                                GUn.add_edge(w1, w2, weight=distance_weight)
                        
                        distance_weight -= 1

        self.origGraph = G
        self.origUnGraph = GUn
         
        self.stats["topn_nodes"] = self.topn_nodes
        self.stats["topn_comms"] = self.topn_comms
        self.stats["origNodes"] = self.origGraph.number_of_nodes()
        self.stats["origEdges"] = self.origGraph.number_of_edges()
        self.stats["origUnNodes"] = self.origUnGraph.number_of_nodes()
        self.stats["origUnEdges"] = self.origUnGraph.number_of_edges()
        self.stats["origDensity"] = nx.density(self.origGraph)
        self.stats["origUnDensity"] = nx.density(self.origUnGraph)
        self.stats["max_nodes"] = self.max_nodes
        if (self.origUnGraph.number_of_nodes() > 0):
            av_degree = float(sum(dict(self.origUnGraph.degree()).values())) / float(self.origUnGraph.number_of_nodes())
        else: 
            av_degree = 0
        if (self.stats["origUnNodes"] > 0): 
            self.stats["origUnAvDegree"] = float(float(self.stats["origUnEdges"]) / float(self.stats["origUnNodes"]))       
        else:
            self.stats["origUnAvDegree"] = 0
        self.stats["origUnAvWeightedDegree"] = av_degree        
        # print(self.origGraph.nodes())
        # print(self.origGraph.edges(data=True))
         # print(dict(self.origGraph.degree()).values())

    def cutoff_graph_max_nodes(self):
        """Use max_nodes to create a cutoff graph for further computations.
        """
        # get the number of edges per note
        nodes = []
        degrees = []
        for n in self.origGraph.nodes():
            nodes.append(n)
            degrees.append(self.origGraph.degree(n, weight='weight'))

        
        nodesUn = []
        degreesUn = []
        for n in self.origUnGraph.nodes():
            nodesUn.append(n)
            degreesUn.append(self.origUnGraph.degree(n, weight='weight'))    
        
        degree_df = pd.DataFrame({"node": nodes, "degree": degrees})
        degree_df = degree_df.sort_values("degree", ascending=False)
        
        # print(degree_df) degree of a directed graph

        cutoff_nodes = list(degree_df.iloc[:self.max_nodes]["node"].values)

        degree_un_df = pd.DataFrame({"node": nodesUn, "degree": degreesUn})
        degree_un_df = degree_un_df.sort_values("degree", ascending=False)
        cutoff_un_nodes = list(degree_un_df.iloc[:self.max_nodes]["node"].values)
        # print(degree_un_df) degree of the undirected graph

        # create the new cutoff graph
        self.cutoffGraph = self.origGraph.subgraph(cutoff_nodes)

        self.cutoffUnGraph = self.origUnGraph.subgraph(cutoff_un_nodes)

        self.stats["cutoffNodes"] = self.cutoffGraph.number_of_nodes()
        self.stats["cutoffEdges"] = self.cutoffGraph.number_of_edges()
        self.stats["cutoffDensity"] = nx.density(self.cutoffGraph)
        if (self.cutoffGraph.number_of_nodes() > 0):
            av_degree = float(sum(dict(self.cutoffGraph.degree()).values())) / float(self.cutoffGraph.number_of_nodes())
        else:
            av_degree = 0
        if (self.stats["cutoffNodes"] > 0):
            self.stats["cutoffAvDegree"] = float(float(self.stats["cutoffEdges"]) / float(self.stats["cutoffNodes"]))     
        else:
            self.stats["cutoffAvDegree"] = 0            
        self.stats["cutoffAvWeighedDegree"] = av_degree

    def write_graph_to_gexf_file(self):
        """Writes the original networkx graph before max_nodes cutoff to gexf-file in paramater gexf_path.
        """
        my_path = os.path.dirname(__file__)

        filename = Path(self.textname.rpartition("/")[2])
        no_ext = filename.with_suffix('')
        new_fname = str(no_ext) + ".graphml"
        self.stats["gexf_file"] = new_fname

        filepath = os.path.join(my_path, self.gexf_path, new_fname)
        self.stats["gexf_loc"] = filepath.rpartition("/")[0] + "/"

        if os.path.isdir(self.gexf_path):
            nx.write_graphml(self.origGraph, self.gexf_path + new_fname)
            nx.write_graphml(self.cutoffGraph, self.gexf_path + "_cutoff_" + new_fname)
            return print("Stored graph as {} in {}.".format(new_fname, self.stats["gexf_loc"]))  # TODO: logging
        else:
            os.mkdir(self.gexf_path)
            print("Created dir {}".format(self.gexf_path))  # TODO: logging
            nx.write_gexf(self.origGraph, self.gexf_path + new_fname)
            nx.write_gexf(self.cutoffGraph, self.gexf_path + "_cutoff_" + new_fname)
            return print("Stored graph as {} in {}.".format(new_fname, self.stats["gexf_loc"]))  # TODO: logging

    def communities_and_betweenness_centrality(self):
        """Compute Louvain communities and betweenness centrality for each node."""

        # communities using louvain-igraph https://louvain-igraph.readthedocs.io/en/latest/intro.html
        # set up for louvain-igraph, use iGraph now after dynamically created netwirk with networkx
        
        # we use undirected graph for community detection as louvain only takes this as input
        
        g2 = ig.Graph.Weighted_Adjacency((nx.to_numpy_matrix(self.cutoffUnGraph, weight="weight")).tolist(),
                                         attr="weight")
        g2.vs["name"] = list(self.cutoffUnGraph.nodes)  # Pray to God they are returned in the same order/ LMAO verificar como muda isso para outros algoritmos de comunidades do networkx
        self.igraph = g2
        # print(g2.is_weighted()) # check if weighted 
        # print(g2.get_edgelist()) # make a list of edges
        # print(g2.es["weight"]) # make a list of weights for each edge 

        if 'weight' in self.igraph.es.attributes():
            self.partitions = louvain.find_partition(g2, louvain.ModularityVertexPartition, weights='weight')
        elif len(self.cutoffUnGraph.nodes) > 0:
            self.partitions = louvain.find_partition(g2, louvain.ModularityVertexPartition)
        else:
            self.partitions = []
        louvaincom = community.best_partition(self.cutoffUnGraph,weight='weight')

        #print(louvaincom) # check community nodes distribution
        #print(community.modularity(louvaincom, self.cutoffUnGraph, weight='weight')) # thi is if you wanted to use another modulaity algorithm

        for com, wordlist in enumerate(self.partitions):
            for nodeid in wordlist:
                self.cutoffUnGraph.nodes[self.igraph.vs[nodeid]["name"]]["com"] = com
                self.cutoffUnGraph.nodes[self.igraph.vs[nodeid]["name"]]["roles"]=[]
        # print(self.cutoffUnGraph.nodes(data=True)) # check nodes to community assignement

        # modularity
        if (len(self.partitions) > 0):
            self.M = self.partitions.quality()
            #self.M = community.modularity(louvaincom, self.cutoffUnGraph, weight='weight')
        else:
            self.M = 0
        self.stats["modularity"] = self.M
        # print(self.stats["modularity"])
        # Betweenness centraliy calculation

        bc_nodes = [] 
        bc_values = []
        bc_coms = []

        # Is the graph directed or nondirected?

        if self.bc_directed:
            self.finalGraph = self.cutoffGraph

            for com, wordlist in enumerate(self.partitions):
                for nodeid in wordlist:
                    self.finalGraph.nodes[self.igraph.vs[nodeid]["name"]]["com"] = com            
        else:
            self.finalGraph = self.cutoffUnGraph

        # prepare Graph for BC — change weights because of the issues in networkx algorithm where for BC the higher the weight is, the weaker is the connection. but not for community detection where it is the opposite
        # https://github.com/networkx/networkx/issues/3369 for more info
        

        for u,v,d in self.finalGraph.edges(data=True):
            d['roles'] = []
            d['weight'] = 1 / d['weight']
            
        # betweenness centrality Cutoff calculation
        bc = nx.betweenness_centrality(self.finalGraph,weight="weight")
        
        # ascribe values to the graph
            
        for n in self.finalGraph.nodes:
            bc_nodes.append(n)
            bc_values.append(bc[n])
            bc_coms.append(self.finalGraph.nodes[n]["com"])

        # create Pandas dataframe for further processing
        bc_df = pd.DataFrame({"node": bc_nodes, "bc": bc_values, "community": bc_coms})
        bc_df = bc_df.sort_values("bc", ascending=False)

        # normalize BC
        if (len(bc_df) > 0):
            bc_float = bc_df[['bc']].values.astype('float')
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(bc_float)
            df_normalized = pd.DataFrame(x_scaled)
            bc_df['bc_norm'] = df_normalized[0].values

        # cut off all BC that is zero
        #bc_df = bc_df[bc_df.bc > 0]

        # create a dataframe for Jenks Breaks processing 
        bc_df_reduced = bc_df
        bc_df_reduced_jenks = bc_df
        bc_df_reduced_jenks_top = bc_df
        self.bc_top_all = bc_df

        for k, v in self.finalGraph.nodes.items():
            self.cutoffUnGraph.nodes[k]["bc"] = bc_df.set_index("node").loc[k]['bc_norm']
            self.finalGraph.nodes[k]["bc"] = bc_df.set_index("node").loc[k]['bc_norm']

        degrees = dict(self.finalGraph.degree)
        degree = []
        for node in self.bc_top_all['node']:
            degree.append(degrees[node])
        degree=pd.Series(degree,name='degree')
        self.bc_top_all=pd.concat([self.bc_top_all,degree],axis=1)
        # # our Jenks breaks works similarly to KMeans, so keeping it here for checking the results
        # km = KMeans(n_clusters=8).fit(bc_df_reduced[['bc']])
        # bc_df_reduced.loc[:,'cluster'] = km.labels_
        # print(bc_df_reduced)
        # print(bc_df_reduced['bc_norm'].median())
        # print(bc_df_reduced['bc'].median())

        # we will now cluster BC nodes into two groups: relevant for our analysis and not
        # usually this should happen at the break (elbow) of the BC curve
        # using Jenks Breaks algorithm, which is also applied in InfraNodus
        # https://stackoverflow.com/a/42375180/712347
        # We use this to choose the top BC nodes to later analyze their distribution

        # create a list of Betweenness Centrality measures for all the nodes
        bc_x = list(bc_df_reduced_jenks['bc'])
        
        # identify the breaking point
        # Jenks Breaks works a bit like kMeans but it identifies the breaks in partition where 
        # values change is relatively high
        if (len(bc_x) > 0):
            breaking_point = get_jenks_breaks(bc_x, 8)
        else:
            breaking_point = 0

        # Creating group for the bc column
        def assign_cluster(bc):
            if bc <= breaking_point[4]:
                return 0
            elif breaking_point[4] < bc <= breaking_point[5]:
                return 1 
            elif breaking_point[5] < bc <= breaking_point[6]:
                return 2     
            else:
                return 3 
                
        # get the very top BC nodes         
        bc_df_reduced_jenks.loc[:,'cluster'] = bc_df_reduced_jenks['bc'].apply(assign_cluster)
    
        # bc_top contains only the very top nodes
        bc_top = bc_df_reduced_jenks[bc_df_reduced_jenks['cluster']>=2]

        bc_top_top = bc_df_reduced_jenks[bc_df_reduced_jenks['cluster']==3]

        # bc_top_lax contains a longer list of top nodes
        bc_top_lax = bc_df_reduced_jenks[bc_df_reduced_jenks['cluster']>=1]

        # let's now make an analysis of how fair the distribution is 
        # for this we split the BC nodes into two groups and see where the border lies
        if (len(bc_x) > 0):
            breaking_point_top = get_jenks_breaks(bc_x, 2)
        else:
            breaking_point = 0

         # Creating group for the bc column
        def assign_cluster_top(bc):
            if bc <= breaking_point_top[1]:
                return 0
            else:
                return 1
        
        bc_df_reduced_jenks_top.loc[:,'cluster'] = bc_df_reduced_jenks_top['bc'].apply(assign_cluster_top)

        bc_top_split = bc_df_reduced_jenks_top[bc_df_reduced_jenks_top['cluster']==1]

        # print(get_jenks_breaks(bc_x, 2)) # just to check what the breaks are
        # print(get_jenks_breaks(bc_x, 3)) 
        # print(get_jenks_breaks(bc_x, 8)) 
        # print(bc_top_top)
        # print(bc_top)
        # print(bc_top_lax)
        # print(bc_top_split)
        # print(bc_df_reduced_jenks)
        self.bc_top = bc_top
        self.bc_top_lax = bc_top_lax
        self.bc_top_split = bc_top_split
        

    def discourse_diversity(self):
        """Computes values E, M, G, C and bias index.

        If plot=True, also prints a matplotlib chart showing topn_nodes distribution in topn communities.
        """
        # entropy
        nodes = []
        bcs = []
        coms = []
            
        for n, data in self.finalGraph.nodes(data=True):
            nodes.append(n)
            bcs.append(data["bc"])
            coms.append(data["com"])

        df = pd.DataFrame({"node": nodes, "bc": bcs, "com": coms})  # Because Pandas DataFrames are cool
        nodes_df = df.sort_values("bc", ascending=False)
        self.nodes_df = nodes_df  # store the original graph

        self.stats["communities"] = len(nodes_df.com.value_counts())

        top_nodes = nodes_df  # If topn_nodes is preferred .ìloc[[:self.topn_nodes]

        top_comms = nodes_df.com.value_counts()  # If topn_nodes is preferred .ìloc[:self.topn_nodescomms]

        # Count topn nodes in topn communities
        cnt = Counter()  # keeps values sorted, as opposed to dictionary
        for com_number in list(top_comms.index):
            cnt[com_number] = 0
            for word in top_nodes.node.values:
                if word in df[df["com"] == com_number]["node"].values:
                    cnt[com_number] += 1

        ord_dict = OrderedDict()
        for community_no, count in cnt.most_common():  # self.topn_comms if preferred
            ord_dict[community_no] = count

        coms = []
        counts = []
        for com, count in cnt.most_common():  # defaults to all, if topn is preferred self.topn_comms
            coms.append(com)
            counts.append(count)

        topn_dist_df = pd.DataFrame({"count": counts}, index=coms)
        
        # print(topn_dist_df) # how many nodes in every community

        # print(self.bc_top_split['community'])

        TSN = len(self.bc_top_split['community']) # Total Split Nodes (top influential nodes Jenks=2)
        E_split = eta(list(self.bc_top_split['community']))
        E_split_ideal = eta(list(range(1, len(self.bc_top_split['community']) + 1)))
        self.stats["TopBCNodesInComm"] = TSN
    
        # entropy ration for Split Nodes
        if E_split_ideal > 0:
            ES = E_split/E_split_ideal
        else:
            ES = 0
    
        # proportion of the top community in BC nodes
        rs_mode = self.bc_top_split['community'].mode()
        if len(rs_mode) > 0:
            most_freq_com_SN = self.bc_top_split['community'].mode()[0]
            RSN = len(self.bc_top_split[self.bc_top_split['community']==most_freq_com_SN])/len(self.bc_top_split['community'])
        else:
            RSN = 0
        self.stats["TopBCNodesProp"] = RSN



        TTN = len(self.bc_top['community']) # Total Top Nodes
        E_top = eta(list(self.bc_top['community']))
        E_top_ideal = eta(list(range(1, len(self.bc_top['community']) + 1)))
        self.stats["BCNodesInComm"] = TTN
        
        # entropy for Top Nodes
        if E_top_ideal > 0:
            ET = E_top/E_top_ideal
        else:
            ET = 0

        # proportion of the top community in BC nodes
        rt_mode = self.bc_top['community'].mode()
        if len(rt_mode) > 0:
            most_freq_com_TN = self.bc_top['community'].mode()[0]
            RTN = len(self.bc_top[self.bc_top['community']==most_freq_com_TN])/len(self.bc_top['community'])
        else:
            RTN = 0
        self.stats["BCNodesProp"] = RTN


        TLN = len(self.bc_top_lax['community'])
        E_top_lax = eta(list(self.bc_top_lax['community']))
        E_top_lax_ideal = eta(list(range(1, len(self.bc_top_lax['community']) + 1)))
        self.stats["BCNodesLaxInComm"] = TLN


        if E_top_lax_ideal > 0:
            EL = E_top_lax/E_top_lax_ideal
        else:
            EL = 0

        # proportion of the top community in BC nodes
        rl_mode = self.bc_top_lax['community'].mode()
        if len(rl_mode) > 0:
            most_freq_com_LN = self.bc_top_lax['community'].mode()[0]
            RLN = len(self.bc_top_lax[self.bc_top_lax['community']==most_freq_com_LN])/len(self.bc_top_lax['community'])
        else:
            RLN = 0
        self.stats["BCNodesLaxProp"] = RLN


        self.stats["entropyTopFirst"] = ES
        self.stats["entropyTop"] = ET
        self.stats["entropyTopLax"] = EL

        E = EL

        # percentage of nodes in largest community
        nr_of_nodes = 0
        for part in self.partitions:
            for node in part:
                nr_of_nodes += 1

        if (len(self.partitions) > 0):
            C = len(self.partitions[0]) / nr_of_nodes
        else: 
            C = 0
        self.C = C
        self.stats["nodesInTopCom"] = C

        # how many of the top BC nodes in the top component
        sn_in_top_c = 0
        tn_in_top_c = 0
        ln_in_top_c = 0
        
        if (len(self.partitions) > 0):
            for node in self.partitions[0]:
                if ((self.bc_top_split).index == node).any():
                    sn_in_top_c += 1
                if ((self.bc_top).index == node).any():
                    tn_in_top_c += 1
                if ((self.bc_top_lax).index == node).any():
                    ln_in_top_c += 1

            
        # number of nodes from the top comm in BC to total length of inf nodes
        if len(self.bc_top_split) > 0:
            BCST = sn_in_top_c/len(self.bc_top_split)
        else: 
            BCST = 0

        if len(self.bc_top) > 0: 
            BCTT = tn_in_top_c/len(self.bc_top)
        else:
            BCTT = 0

        if len(self.bc_top_lax) > 0:
            BCLT = ln_in_top_c/len(self.bc_top_lax)
        else: 
            BCLT = 0

        # percentage of nodes in the giant component
        if (nr_of_nodes > 0):
            G = len(self.igraph.components().giant().vs) / nr_of_nodes
        else:
            G = 0
        self.G = G
        self.stats["nodesInGiantCom"] = G

        M = self.M

        if nr_of_nodes == 0:
            self.stats["biasIndex"] = "Dispersed"

        elif TSN <= 4: # very few top BC nodes, so influence distribution is biased – alert!
            
            if ES == 0 and G > 0.5 and TSN != 1: # they all belong to the same community and > 50% of the graph is connected

                # as ES == 0, all the top BC nodes belong to only 1 community, that's why BCST is either 0 or 1

                if TSN == 0:
                    if (self.cutoffGraph.number_of_nodes() > 0):
                        if G < 0.5:
                            self.stats["biasIndex"] = "Dispersed"
                        else:
                            if M > 0.65: # pronounced comm, several center of influence, one is focused but the rest is dispersed
                                self.stats["biasIndex"] = "Dispersed"                     
                            elif 0.4 <= M <= 0.65: # while community structure may be high, all he most inf nodes are in the biggest comm
                                self.stats["biasIndex"] = "Polarized" #TODO may need modification
                            elif 0.2 <= M < 0.4: # low comm, influence center with all top BC nodes in it
                                self.stats["biasIndex"] = "Focused"
                            elif M < 0.2: # low comm, influence center with all top BC nodes in it
                                self.stats["biasIndex"] = "Biased" 
                    else:
                        self.stats["biasIndex"] = "Dispersed"   
                
                elif C > 0.5 and BCST == 1: # > 50% of nodes in the top comm and all top BC belong to it
                    
                    self.stats["biasIndex"] = "Biased" # everything is towards the same narrative
                
                elif C > 0.5 and BCST == 0: # > 50% of nodes in top comm, but all the influential words in another comm
                    
                    if M > 0.4: # pronounced comm, BC nodes oppose the top C
                        self.stats["biasIndex"] = "Polarized"  # TODO interesting idea here                    
                    elif 0.2 > M < 0.4: # not pronounced comm, BC nodes support the top C
                        self.stats["biasIndex"] = "Focused"
                    elif M < 0.2: #low comm, 2 centers of influence, super linked
                        self.stats["biasIndex"] = "Biased"
               
                elif C <= 0.5 and BCST == 1: # less than half (but signif num) are in top C as well as all the top BC

                    if M > 0.65: # pronounced comm, several center of influence, one is focused but the rest is dispersed
                        self.stats["biasIndex"] = "Dispersed"                     
                    elif 0.4 < M <= 0.65: # while community structure may be high, all he most inf nodes are in the biggest comm
                        self.stats["biasIndex"] = "Polarized" #TODO may need modification
                    elif M < 0.4: # low comm, influence center with all top BC nodes in it
                        self.stats["biasIndex"] = "Biased"

                elif C <= 0.5 and BCST == 0: # less than half in top (signif) comm but ALL top BC nodes are in another
                     
                    if M > 0.65: # very pronounced community structure, 2 centers of influence
                        self.stats["biasIndex"] = "Diversified" 
                    elif 0.65 > M >= 0.4: # pronounced comm, 2 centers of influence, connected
                        self.stats["biasIndex"] = "Polarized" #TODO interesting idea here                     
                    elif 0.2 < M < 0.4: # not pronounced comm, 2 centers of influence, linked
                        self.stats["biasIndex"] = "Focused"
                    elif M < 0.2: #low comm, 2 centers of influence, super linked
                        self.stats["biasIndex"] = "Biased"                        

            elif ES == 0 and G < 0.5 and TSN != 1: # all the nodes are in the same community but less than half nodes are in G

                if M > 0.65:
                    self.stats["biasIndex"] = "Dispersed" 
                elif M > 0.4:
                    self.stats["biasIndex"] = "Focused" 
                elif M <= 0.4:
                    self.stats["biasIndex"] = "Biased" 
                    
            elif ES >= 0.75 and TSN > 2:

                if M > 0.65:
                    self.stats["biasIndex"] = "Dispersed"
                elif 0.4 <= M <= 0.65:
                    self.stats["biasIndex"] = "Diversified"
                elif 0.2 < M < 0.4:
                    self.stats["biasIndex"] = "Focused"
                else:
                    self.stats["biasIndex"] = "Biased"
            
            elif (ES < 0.75 and ES > 0) or (ES == 0 and TSN == 1) or (TSN <= 2 and ES >= 0.75): # either it's ab aab, aabb or aaab or a
                
                # here we analyze like for TSN > 4 but with a bigger number of nodes
                if (TTN > 4) or (2 < TTN <= 4 and TSN == 1): #means influence is relatively spread

                    if ET == 0 and G < 0.5: # all the nodes are in the same community but less than half nodes are in G
                        
                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed" 
                        if M > 0.4:
                            self.stats["biasIndex"] = "Focused" 
                        elif M <= 0.4:
                            self.stats["biasIndex"] = "Biased" 

                    elif ET == 0 and G > 0.5:

                        if BCTT == 1: # all the top nodes (and quite a few of them!) are in the tom comm
                        
                            if M >= 0.4: # while community structure may be high, all he most inf nodes are in the biggest comm
                                self.stats["biasIndex"] = "Focused"
                            elif M < 0.4: # low comm, influence center with all top BC nodes in it
                                self.stats["biasIndex"] = "Biased"

                        elif BCTT == 0: # less than half in top (signif) comm but ALL top BC nodes are in ANOTHER
                            
                            if M > 0.65: # very pronounced community structure, 2 centers of influence
                                self.stats["biasIndex"] = "Diversified" 
                            elif 0.65 > M >= 0.4: # pronounced comm, 2 centers of influence, connected
                                self.stats["biasIndex"] = "Polarized" #TODO interesting idea here                     
                            elif 0.2 < M < 0.4: # not pronounced comm, 2 centers of influence, linked
                                self.stats["biasIndex"] = "Focused"
                            elif M < 0.2: #low comm, 2 centers of influence, super linked
                                self.stats["biasIndex"] = "Biased"   
                        
                        
                    
                    if ET > 0.42 and RTN <= 0.5: # a few nodes but most of them are not in top comm
                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed"
                        elif 0.4 <= M <= 0.65:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"

                    elif ET > 0.42 and RTN > 0.5: # there's quite a few nodes, most of them are in the same comm, but the rest is quite dispersed

                        if M > 0.65 and BCTT < 0.5: # most nodes are not in top comm
                            self.stats["biasIndex"] = "Dispersed"
                        elif M >= 0.4 and BCTT <= 0.8:
                            self.stats["biasIndex"] = "Diversified"
                        elif M > 0.2 and BCTT <= 0.8:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"

                    elif (ET <= 0.42 and RTN > 0.5) or (RTN > 0.75): # there's a few nodes , most of them are in the same com, the rest is also homogeneous
                        
                        if M > 0.65:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.65 >= M > 0.4:
                            self.stats["biasIndex"] = "Polarized" #TODO interesting idea here
                        elif M < 0.4:
                            self.stats["biasIndex"] = "Biased"
                        
                    elif (ET <= 0.42 and RTN <= 0.5): # there's a few nodes, most of them not in the top comm, the rest is homogeneous

                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed"
                        elif 0.65 >= M > 0.4:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"
                
                # we had too few top nodes, so we expanded the search. means influence is very well spread
                elif (TLN > 4) or (TLN < 4 and TLN >= TSN):

                    if EL > 0.42 and RLN <= 0.5: # a few nodes but most of them are not in top comm
                
                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed"
                        elif 0.4 <= M <= 0.65:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"

                    elif EL > 0.42 and RLN > 0.5: # there's quite a few nodes, most of them are in the same comm, but the rest is quite dispersed

                        if M > 0.65 and BCLT < 0.5: # most nodes are not in top comm
                            self.stats["biasIndex"] = "Dispersed"
                        elif M >= 0.4 and BCLT <= 0.8:
                            self.stats["biasIndex"] = "Diversified"
                        elif M > 0.2 and BCLT <= 0.8:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"

                    elif (EL <= 0.42 and RLN > 0.5) or (RLN > 0.75): # there's a few nodes , most of them are in the same com, the rest is also homogeneous
                        
                        if M > 0.65:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.65 >= M > 0.4:
                            self.stats["biasIndex"] = "Polarized" #TODO interesting idea here
                        elif M < 0.4:
                            self.stats["biasIndex"] = "Biased"
                        
                    elif (EL <= 0.42 and RLN <= 0.5): # there's a few nodes, most of them not in the top comm, the rest is homogeneous

                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed"
                        elif 0.65 >= M > 0.4:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"                   

                else: # didn't find 
                    if RSN > 0.5: # aaab or aab
                        # we didn't find any other separation, so probably these N nodes are outliers
                        if M > 0.65:
                            if BCST > 0.5: #dispersed community but most BC nodes are in top C                                
                                self.stats["biasIndex"] = "Diversified"
                            elif BCST <= 0.5:
                                self.stats["biasIndex"] = "Dispersed"
                        elif 0.4 <= M <= 0.65:
                            if BCST > 0.5:
                                self.stats["biasIndex"] = "Focused"
                            elif BCST <= 0.5:
                                self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            if BCST > 0.5:
                                self.stats["biasIndex"] = "Biased"
                            elif BCST <= 0.5:
                                self.stats["biasIndex"] = "Focused"
                        else:
                            self.stats["biasIndex"] = "Biased"

                    elif RSN <= 0.5: #ab #aabb
                        #TODO interesting idea here
                        if M > 0.65:
                            self.stats["biasIndex"] = "Dispersed"
                        elif 0.4 <= M <= 0.65:
                            self.stats["biasIndex"] = "Diversified"
                        elif 0.2 < M < 0.4:
                            self.stats["biasIndex"] = "Focused"
                        elif M < 0.2:
                            self.stats["biasIndex"] = "Biased"
            



        
        elif TSN > 4:

            if ES == 0 and G < 0.5: # all the nodes are in the same community but less than half nodes are in G
                
                if M > 0.65:
                    self.stats["biasIndex"] = "Dispersed" 
                if M > 0.4:
                    self.stats["biasIndex"] = "Focused" 
                elif M <= 0.4:
                    self.stats["biasIndex"] = "Biased" 

            elif ES == 0 and G > 0.5:

                if BCST == 1: # all the top nodes (and quite a few of them!) are in the tom comm
                  
                    if M >= 0.4: # while community structure may be high, all he most inf nodes are in the biggest comm
                        self.stats["biasIndex"] = "Focused"
                    elif M < 0.4: # low comm, influence center with all top BC nodes in it
                        self.stats["biasIndex"] = "Biased"

                if BCST == 0: # less than half in top (signif) comm but ALL top BC nodes are in ANOTHER
                     
                    if M > 0.65: # very pronounced community structure, 2 centers of influence
                        self.stats["biasIndex"] = "Diversified" 
                    elif 0.65 > M >= 0.4: # pronounced comm, 2 centers of influence, connected
                        self.stats["biasIndex"] = "Polarized" #TODO interesting idea here                     
                    elif 0.2 < M < 0.4: # not pronounced comm, 2 centers of influence, linked
                        self.stats["biasIndex"] = "Focused"
                    elif M < 0.2: #low comm, 2 centers of influence, super linked
                        self.stats["biasIndex"] = "Biased"   
            
            elif ES > 0.42 and RSN <= 0.5:
                
                if M > 0.65:
                    self.stats["biasIndex"] = "Dispersed"
                elif 0.4 <= M <= 0.65:
                    self.stats["biasIndex"] = "Diversified" #TODO interesting thing here
                elif 0.2 < M < 0.4:
                    self.stats["biasIndex"] = "Focused"
                else:
                    self.stats["biasIndex"] = "Biased"

            elif ES > 0.42 and RSN > 0.5:

                if M > 0.65 and BCST < 0.5:
                    self.stats["biasIndex"] = "Dispersed"
                elif M >= 0.4 and BCST <= 0.8:
                    self.stats["biasIndex"] = "Diversified"
                elif M > 0.2 and BCST <= 0.8:
                    self.stats["biasIndex"] = "Focused"
                else:
                    self.stats["biasIndex"] = "Biased"

            elif (ES <= 0.42 and RSN > 0.5) or (RSN > 0.75):
                
                if M > 0.65:
                    self.stats["biasIndex"] = "Diversified"
                elif 0.65 >= M > 0.4:
                    self.stats["biasIndex"] = "Focused"
                elif M < 0.4:
                    self.stats["biasIndex"] = "Biased"
                
            elif (ES <= 0.42 and RSN <= 0.5):

                if M > 0.65:
                    self.stats["biasIndex"] = "Dispersed"
                elif 0.65 >= M > 0.4:
                    self.stats["biasIndex"] = "Diversified"
                elif 0.2 < M < 0.4:
                    self.stats["biasIndex"] = "Focused"
                else:
                    self.stats["biasIndex"] = "Biased"
  
        # END OF BIAS INDEX CALCULATION

        self.biasIndex = self.stats["biasIndex"]

        if self.biasIndex == 'Biased':
            print("{} bias index: BIASED".format(self.textname))
        elif self.biasIndex == 'Focused':
            print("{} bias index: FOCUSED".format(self.textname))
        elif self.biasIndex == 'Diversified':
            print("{} bias index: DIVERSIFIED".format(self.textname))
        elif self.biasIndex == 'Polarized':
            print("{} bias index: POLARIZED".format(self.textname))
        elif self.biasIndex == 'Dispersed':
            print("{} bias index: DISPERSED".format(self.textname))


        # Possible plotting, in notebook add ``%matplotlib inline`` and ``import matplotlib.pyplot as plt``
        if (nr_of_nodes > 0):
            if self.plot:
                print('plotting ...')
                xticks = [str(i) for i in list(topn_dist_df.index)]
                ax = topn_dist_df.plot(
                    title="Node-distribution in communities.\n{} entropy:{:.3f}".format(
                        self.cutoffGraph.graph['name'],
                        E), use_index=False)
                ax.set_xlabel("Community number\n")
                ax.set_ylabel("Words (nodes) per community.")
                ax.set_xticks(np.arange(len(xticks)))
                ax.set_xticklabels(xticks)
                plt.show()

        set_junctions_hubs(self, how_many_hubs = 3, how_many_junct = 3)
        set_edges_by_nodes(self)


    def generate_stats_dataframe(self):
        """Collect settings and statistics from process and transform to a Pandas DataFrame object for further analysis.

        :return: Pandas DataFrame object.
        """

        return pd.DataFrame.from_dict(self.stats, orient="index", columns=[self.cutoffGraph.name])

    
def get_jenks_breaks(data_list, number_class):
        data_list.sort()
        mat1 = []
        for i in range(len(data_list) + 1):
            temp = []
            for j in range(number_class + 1):
                temp.append(0)
            mat1.append(temp)
        mat2 = []
        for i in range(len(data_list) + 1):
            temp = []
            for j in range(number_class + 1):
                temp.append(0)
            mat2.append(temp)
        for i in range(1, number_class + 1):
            mat1[1][i] = 1
            mat2[1][i] = 0
            for j in range(2, len(data_list) + 1):
                mat2[j][i] = float('inf')
        v = 0.0
        for l in range(2, len(data_list) + 1):
            s1 = 0.0
            s2 = 0.0
            w = 0.0
            for m in range(1, l + 1):
                i3 = l - m + 1
                val = float(data_list[i3 - 1])
                s2 += val * val
                s1 += val
                w += 1
                v = s2 - (s1 * s1) / w
                i4 = i3 - 1
                if i4 != 0:
                    for j in range(2, number_class + 1):
                        if mat2[l][j] >= (v + mat2[i4][j - 1]):
                            mat1[l][j] = i3
                            mat2[l][j] = v + mat2[i4][j - 1]
            mat1[l][1] = 1
            mat2[l][1] = v
        k = len(data_list)
        kclass = []
        for i in range(number_class + 1):
            kclass.append(min(data_list))
        kclass[number_class] = float(data_list[len(data_list) - 1])
        count_num = number_class
        while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
            idx = int((mat1[k][count_num]) - 2)
            # print "val = " + str(data_list[idx])
            kclass[count_num - 1] = data_list[idx]
            k = int((mat1[k][count_num] - 1))
            count_num -= 1
        return kclass

def eta(data, unit='shannon'):
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    ent = 0

    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent
