from .vis import partitiontodict
from .discourse_functions import *
import numpy as np

class Discourse:

    def __init__(self,
                 textnet,
                 node_roles=['junction','hub'],
                 how_many_junct = 3,
                 how_many_hubs = 3):


        self.discourse = self._setup_roles(textnet,how_many_junct=how_many_junct,how_many_hubs=how_many_hubs, node_roles=node_roles)

    def _setup_roles(self,textnet, **kwargs):
        '''
         Extracts nodes inside communities with higher betweeness centrality and connectivity.
         Setup node role information in given textnet.
        '''
        rolearg = dict()
        for arg in kwargs:
            if arg == 'node_roles':
                names=kwargs['node_roles']

            if arg == 'how_many_hubs':
                rolearg['hub']=kwargs['how_many_hubs']
                setattr(textnet,'n_hub',rolearg['hub'])

            if arg == 'how_many_junct':
                rolearg['junction']=kwargs['how_many_junct']
                setattr(textnet,'n_junction',rolearg['junction'])

        roles=dict()
        for i in range(len(names)):
            roles[names[i]]=list()
            for com,df in textnet.bc_top_all.groupby('community'):
                print('Community {}:'.format(com))
                roles[names[i]].append(dconfig[names[i]](textnet,df,rolearg[names[i]]))

        #TODO- Verificar condicao 
        allothers=[]
        for i in range(max(textnet.bc_top_all['community'])+1):
            allothers.append([])
        for node in textnet.finalGraph.nodes:
            if not np.array([names[i] in textnet.finalGraph.nodes[node]["roles"] for i in range(len(names))]).all():
                allothers[textnet.finalGraph.nodes[node]['com']].append(node)

        return roles,allothers

#TODO- Consider to take junction and hub configurations out of setup roles, put in a lookup table for configurable function

    @property
    def junctions(self):
        return self.discourse[0]['junction']

    @property
    def hubs(self):
        return self.discourse[0]['hub']

    @property
    def others(self):
        return self.discourse[1]

    @property
    def info(self):
        print('Edge types:\n\n')
        print('hub to node\n')
        print('inter hub\n')
        print('community intern\n')
        print('community intern hub junction\n')
        print('inter junction\n')
        print('junction diffusor\n')
        print('inter community\n')

# TODO- Consider expand discourse function by using class decorators in further refactoration


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
