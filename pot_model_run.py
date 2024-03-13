import vart
from vart import Runner
import xir
import vitis_ai_library

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
    # create graph runner
    xmodel_file = "result/linregress.xmodel"
    graph = xir.Graph.deserialize(xmodel_file)
    runner = vitis_ai_library.GraphRunner.create_graph_runner(graph)
    # get input and output tensor buffers
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()
    # run graph runner
    v = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
    runner.wait(v)
    output_data = np.asarray(output_tensor_buffers[0])

