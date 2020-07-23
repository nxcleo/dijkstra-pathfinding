import networkx as nx
import matplotlib.pyplot as plt

g = nx.read_weighted_edgelist('basicGraph.txt', create_using=nx.Graph(), nodetype=int)

print(nx.info(g))

pos=nx.spring_layout(g)

nx.draw(g, with_labels=True, pos=pos, node_size = 0)
edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in g.edges(data=True)])
nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels, font_size=6)

plt.figure(figsize=(16,9))
plt.show()