from .vis import partitiontodict

def setup_roles(textnet, how_many_junct = 3, how_many_hubs = 3):
    '''
     Extracts nodes inside communities with higher betweeness centrality and connectivity.
     Setup node role information in given textnet.
    
    '''

    setattr(textnet,'n_hub',how_many_hubs)
    setattr(textnet,'n_junction',how_many_junct)
    
    alljnames=[]
    allhnames=[]
    for com,df in textnet.bc_top_all.groupby('community'):
        jnames=df.sort_values(['bc_norm'],axis=0,ascending=False)['node'].iloc[0:how_many_junct]
        hnames=df.sort_values(['degree'],axis=0,ascending=False)['node'].iloc[0:how_many_hubs]
        for name in jnames:
            textnet.finalGraph.nodes[name]["roles"].append("junction")
        for name in hnames:
            textnet.finalGraph.nodes[name]["roles"].append("hub")
        alljnames.append(list(jnames))
        allhnames.append(list(hnames))

    allothers=[]
    for i in range(max(textnet.bc_top_all['community'])+1):
        allothers.append([])
    for node in textnet.finalGraph.nodes:
        if ("junction" not in textnet.finalGraph.nodes[node]["roles"]) and ("hub" not in textnet.finalGraph.nodes[node]["roles"]):
            allothers[textnet.finalGraph.nodes[node]['com']].append(node)

    return alljnames,allhnames,allothers

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
        elif (partition[edge[0]] == partition[edge[1]]) and ((('hub' in textnet.finalGraph.nodes[edge[0]]['roles']) and ('junction' in textnet.finalGraph.nodes[edge[1]]['roles'])) or ( ('junction' in textnet.finalGraph.nodes[edge[0]]['roles']) and ('hub' in
            textnet.finalGraph.nodes[edge[1]]['roles']))):
            edge[2]['roles'].append('community hub junction')

        elif (partition[edge[0]] != partition[edge[1]]) and (('junction' in textnet.cutoffUnGraph.nodes[edge[0]]['roles']) and ('junction' in textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
            edge[2]['roles'].append('inter junction')          
        elif (partition[edge[0]] != partition[edge[1]]) and (('junction' in textnet.cutoffUnGraph.nodes[edge[0]]['roles']) ^ ('junction' in textnet.cutoffUnGraph.nodes[edge[1]]['roles'])):
            edge[2]['roles'].append('diffusor')

        elif (partition[edge[0]] != partition[edge[1]]):
            edge[2]['roles'].append('inter communities')

def max_path(textnet,edge_function=['inter junction'], num=2):
    '''
    Get num inter junctions edges with highest weight for each junction node
    '''
    max_w = dict()
    for node,config in textnet.finalGraph.nodes.items():
        weights = dict()
        if 'junction' in config['roles']:
            for neigh in textnet.finalGraph.neighbors(node):
                if 'inter junction' in textnet.finalGraph.get_edge_data(node,neigh)['roles']:
                    weights[(node,neigh)]=textnet.finalGraph.get_edge_data(node,neigh)['weight']
            weights = {k:v for k,v in sorted(weights.items(),key=lambda x: x[1],reverse=True)}
            keys = list(weights.keys())[0:num] 
            vals = list(weights.values())[0:num]
            max_w.update({k:(1.0,0.0,0.0,vals[0]) for k,v in zip(keys,vals)})

    return max_w
