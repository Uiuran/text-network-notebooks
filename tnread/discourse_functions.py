def config_junction(textnet,df,how_many_junct):

    jnames=df.sort_values(['bc_norm'],axis=0,ascending=False)['node'].iloc[0:how_many_junct]
    for name in jnames:
        textnet.finalGraph.nodes[name]["roles"].append("junction")
    return list(jnames)

def config_hub(textnet,df,how_many_hubs):

    hnames=df.sort_values(['degree'],axis=0,ascending=False)['node'].iloc[0:how_many_hubs]
    for name in hnames:
        textnet.finalGraph.nodes[name]["roles"].append("hub")
    return list(hnames)

dconfig = dict({
    'junction':config_junction,
    'hub':config_hub})


