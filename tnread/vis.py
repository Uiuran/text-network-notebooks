import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import fa2l

def partitiontodict(partitions):
  d = dict()
  for idx,cluster in enumerate(partitions.partitions):
    for node in cluster:
      d[partitions.partitions._graph.vs['name'][node]] = idx
  return d

def to_numpy(d):
  for k,v in d.items():
    d[k] = np.array(v)
  return d


class TextnetVis:
  
  def __init__(self, textnet):
        
    self.textnet = textnet
    self.partition=partitiontodict(self.textnet)
    self.community_layout()        
    # Set template according to the discourse/betweeness centrality
    self._discourse_template()
    
  def community_layout(self):
      """
      Compute the layout for a modular graph.
      
      Arguments:
      ----------
      g -- networkx.Graph or networkx.DiGraph instance
          graph to plot
  
      partition -- dict mapping int node -> int community
          graph partitions
  
      Returns:
      --------
      pos -- dict mapping int node -> (float x, float y)
          node positions
  
       """ 
      pos_communities = self._position_communities(scale=1.2) 
      pos_nodes = self._position_nodes(scale=1.)
        
      # combine positions
      self.pos = dict()     
      
      for node in self.textnet.cutoffUnGraph.nodes():
           self.pos[node] = pos_communities[node]+pos_nodes[node]     

  def _position_communities(self, **kwargs):
  
      # create a weighted graph, in which each node corresponds to a community,
      # and each edge weight to the number of edges between communities
      between_community_edges = self._find_between_community_edges()
  
      communities = set(self.partition.values())
      hypergraph = nx.Graph()
      hypergraph.add_nodes_from(communities)
      for (ci, cj), edges in between_community_edges.items():
          hypergraph.add_edge(ci, cj, weight=len(edges))
  
      # find layout for communities
      pos_communities = fa2l.force_atlas2_layout(hypergraph,
                                iterations=1000,
                                pos_list=None,
                                node_masses=None,
                                outbound_attraction_distribution=True,
                                lin_log_mode=True,
                                prevent_overlapping=False,
                                edge_weight_influence=.1,
                                jitter_tolerance=1.0,
                                barnes_hut_optimize=True,
                                barnes_hut_theta=0.5,
                                scaling_ratio=10000,
                                strong_gravity_mode=True,
                                multithread=False,
                                gravity=400)
      # set node positions to position of community
      pos = dict()
      for node, community in self.partition.items():
          pos[node] = np.array(pos_communities[community])
  
      return pos

  def _find_between_community_edges(self):
  
      edges = dict()
  
      for (ni, nj) in self.textnet.cutoffUnGraph.edges():
          ci = self.partition[ni]
          cj = self.partition[nj]
  
          if ci != cj:
              try:
                  edges[(ci, cj)] += [(ni, nj)]
              except KeyError:
                  edges[(ci, cj)] = [(ni, nj)]
  
      return edges

  def _position_nodes(self, **kwargs):
      """
      Positions nodes within communities.
      """
  
      communities = dict()
      for node, community in self.partition.items():
          try:
              communities[community] += [node]
          except KeyError:
              communities[community] = [node]
  
      pos = dict()      
      for ci, nodes in communities.items():
          subgraph = self.textnet.cutoffUnGraph.subgraph(nodes)            
          pos_subgraph = fa2l.force_atlas2_layout(subgraph,
                                iterations=1000,
                                pos_list=None,
                                node_masses=None,
                                outbound_attraction_distribution=True,
                                lin_log_mode=False,
                                prevent_overlapping=False,
                                edge_weight_influence=1.0,
                                jitter_tolerance=1.0,
                                barnes_hut_optimize=True,
                                barnes_hut_theta=0.5,
                                scaling_ratio=10000,
                                strong_gravity_mode=True,
                                multithread=False,
                                gravity=400.)
          pos.update(pos_subgraph)
        
          for node in pos:
            pos[node] = np.array(pos[node])
              
      return pos
    
  def _discourse_template(self):
    
    self.node_color=dict()    
    self.node_size=dict()
    self.font_size = dict()
    self.font_color = dict()
    self.templatedgraph = nx.Graph()
    
    for node,node_config in self.textnet.cutoffUnGraph.nodes.items():
      self.node_color[node]= node_config['com']
      self.templatedgraph.add_node(node)
      
      degree=self.textnet.cutoffUnGraph.degree[node]
      node_size_std = 10*degree
    
      if "hub" in node_config["roles"]:
        self.node_size[node] = node_size_std*(1.0+5.0*node_config['bc']) 
        self.font_color[node] = 'blue'
        self.font_size[node]= 25+node_config['bc']*40
    
      if "junction" in node_config["roles"]:
        self.node_size[node] = node_size_std*(1.0+5.0*node_config['bc']) 
        self.font_color[node] = 'red'
        self.font_size[node]= 25+node_config['bc']*40        
    
      if len(node_config["roles"]) == 0:
        self.node_size[node] = node_size_std
        self.font_color[node] = 'black'
        self.font_size[node]= 25+node_config['bc']*40        

    
    self.edge_color=[]

    
    for edge in self.textnet.cutoffUnGraph.edges.data():       
      if self.partition[edge[0]] == self.partition[edge[1]]:
        self.templatedgraph.add_edge(edge[0],edge[1],weight=edge[2]['weight'])
        self.edge_color.append((0.0,0.0,0.0,.5))        

      elif (self.partition[edge[0]] != self.partition[edge[1]]) and (('junction' in self.textnet.cutoffUnGraph.nodes[edge[0]]['roles']) and ('junction' in self.textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
        self.templatedgraph.add_edge(edge[0],edge[1],weight=edge[2]['weight'])
        self.edge_color.append((1.0,0.0,0.0,edge[2]['weight']))
        
      elif (self.partition[edge[0]] != self.partition[edge[1]]) and (('junction' in self.textnet.cutoffUnGraph.nodes[edge[0]]['roles']) ^ ('junction' in self.textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
        pass
        #self.templatedgraph.add_edge(edge[0],edge[1],weight=edge[2]['weight'])
        #self.edge_color.append((0.0,0.0,1.0,edge[2]['weight']))

      elif (self.partition[edge[0]] != self.partition[edge[1]]):
        pass
    
      elif edge[2]['weight'] < 0.09:
        self.templatedgraph.add_edge(edge[0],edge[1],weight=edge[2]['weight'])
        self.edge_color.append((0.5,0.5,0.5,0.05))

  def plot_textnet(self, plotter='matplotlib'):       
    
    if plotter=='matplotlib':
      fig=plt.figure(figsize=(20,20))    
    # Draw networks
      nx.draw(self.templatedgraph, 
              self.pos,             
              node_color=list(self.node_color.values()), edge_color=self.edge_color, 
              node_size=list(self.node_size.values()), width=2.0);     
    # Plot label with different sizes    
      for node, (x, y) in self.pos.items():
        com=self.textnet.bc_top_all.set_index('node').loc[node]['community']
        if 'hub' in self.textnet.cutoffUnGraph.nodes[node]['roles']:
          plt.text(x, y, node, fontsize=self.font_size[node], color=self.font_color[node], ha='center', va='center')
        elif 'junction' in self.textnet.cutoffUnGraph.nodes[node]['roles']:
          plt.text(x, y, node, fontsize=self.font_size[node], color=self.font_color[node], ha='center', va='center')
        else:          
          pass
      plt.show()
    elif plotter=='plotly':
        raise NotImplementedError('Plotly vis still not working.')
    else:
        raise ValueError('Only plotly and matplotlib are accepted as plotter.')
