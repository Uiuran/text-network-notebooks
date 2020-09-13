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
