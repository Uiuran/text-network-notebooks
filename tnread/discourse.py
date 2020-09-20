from .vis import partitiontodict

def set_junctions_hubs(textnet, how_many_junct = 3, how_many_hubs = 3):
    '''
     Extracts nodes inside communities with higher betweeness centrality and connectivity.
     Setup node role information in given textnet.
    
    '''

    setattr(textnet,'n_hub',how_many_hubs)
    setattr(textnet,'n_junction',how_many_junct)

    for com,df in textnet.bc_top_all.groupby('community'):
        jnames=df.sort_values(['bc_norm'],axis=0,ascending=False)['node'].iloc[0:how_many_junct]
        hnames=df.sort_values(['degree'],axis=0,ascending=False)['node'].iloc[0:how_many_hubs]
        for name in jnames:
            textnet.cutoffUnGraph.nodes[name]["roles"].append("junction")
        for name in hnames:
            textnet.cutoffUnGraph.nodes[name]["roles"].append("hub")

def set_edges_by_nodes(textnet):
    '''
     Set edge roles according to the node roles and community membership. If nodes are not configured, then it leaves edge roles as an empty list.
    '''
    partition=partitiontodict(textnet)
    for edge in textnet.finalGraph.edges.data():
        if ('hub' in textnet.finalGraph.nodes[edge[0]]['roles']) or ('hub' in textnet.finalGraph.nodes[edge[1]]['roles']):
            edge[2]['roles'].append('hub')
        elif ('hub' in textnet.finalGraph.nodes[edge[0]]['roles']) and ('hub' in textnet.finalGraph.nodes[edge[1]]['roles']):
            edge[2]['roles'].append('inter hub')
 
        if partition[edge[0]] == partition[edge[1]]:
            edge[2]['roles'].append('community intern')
        elif (partition[edge[0]] == partition[edge[1]]) and (('hub' in textnet.finalGraph.nodes[edge[0]]['roles']) and ('junction' in textnet.finalGraph.nodes[edge[1]]['roles'])) or ( ('junction' in textnet.finalGraph.nodes[edge[0]]['roles']) and ('hub' in textnet.finalGraph.nodes[edge[1]]['roles'])):
            edge[2]['roles'].append('community hub junction')

        elif (partition[edge[0]] != partition[edge[1]]) and (('junction' in textnet.cutoffUnGraph.nodes[edge[0]]['roles']) and ('junction' in textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
            edge[2]['roles'].append('inter junction')          
        elif (partition[edge[0]] != partition[edge[1]]) and (('junction' in textnet.cutoffUnGraph.nodes[edge[0]]['roles']) ^ ('junction' in textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
            edge[2]['roles'].append('diffusor')

        elif (partition[edge[0]] != partition[edge[1]]):
            edge[2]['roles'].append('inter communities') 

def get_hubs(textnet, community=0):
    '''
     Get all hub names from community (must be an integer, if its none get all hubs)
     Hubs are the nodes with highest degree.
    '''
    pass

def get_junctions(textnet, community=0):
    '''
     Get all junction names from community (must be an integer, if its none get all junctions).
     Junctions are the nodes with highest betweeness-centrality.
    '''
    pass
