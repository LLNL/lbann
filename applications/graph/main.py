import utils.walk

graph_file = utils.walk.SnapGraph.download('ego-Facebook')
graph = utils.walk.SnapGraph(graph_file)
graph.walk()
