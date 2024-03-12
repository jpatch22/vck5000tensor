import vart
from vart import Runner
import xir
#import vitisai_runtime as vitis_ai_library
def get_child_subgraph_dpu(graph):
    assert graph is not None, "'graph' should not be None"
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root graph"
    print(root_subgraph, root_subgraph.is_leaf)
    if root_subgraph.is_leaf:
        return [root_subgraph]

    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
            cs
            for cs in child_subgraphs
#            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
            ]

def main():
    model_path = "result/linregress.xmodel"
    
    #print(dir(xir.Graph), "\n")
    #print(dir(vart))
    graph = xir.Graph(model_path)
     
#    subgraphs = get_child_subgraph_dpu(graph)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    print(subgraphs)
    return
    dpu_runners = []
    threads = 1
    for i in range(threads):
        dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    print(dpu_runners)

    return
#    print(dir(graph))
    print(subgraph)
    #runner = vart.GraphRunner.create_graph_runner(graph)

    runner = vart.RunnerExt.create_runner(subgraph, 'run')


    return
    print(dir(vart))
    dpu_runner = vart.Runner.create_runner(model_path, 'run')
    
    print(runner)


if __name__ == "__main__":
    main()
