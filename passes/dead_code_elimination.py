def eliminate_dead_code_fx(graph):
    used = set()
    output_node = next(n for n in reversed(graph.nodes) if n.op == "output")
    queue = list(output_node.all_input_nodes)
    while queue:
        node = queue.pop()
        if node in used:
            continue
        used.add(node)
        queue.extend(node.all_input_nodes)

    for node in list(graph.nodes):
        if node.op != "output" and node not in used:
            graph.erase_node(node)
    return graph