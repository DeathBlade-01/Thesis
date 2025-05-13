import sympy as sp
from collections import defaultdict

def chromatic_polynomial(edges):
    """
    Calculate the chromatic polynomial of a graph specified by its edges.
    
    Args:
        edges: List of tuples representing edges, e.g. [(1,2), (2,3), (1,3)]
    
    Returns:
        A sympy expression representing the chromatic polynomial
    """
    print("Computing chromatic polynomial for graph with edges:", edges)
    
    # Convert edge list to adjacency list
    graph = defaultdict(list)
    vertices = set()
    
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        vertices.add(u)
        vertices.add(v)
    
    # Create symbolic variable
    k = sp.Symbol('k')
    
    # Use deletion-contraction recurrence to compute the polynomial
    return _chromatic_polynomial_recursive(graph, vertices, edges, k)

def _chromatic_polynomial_recursive(graph, vertices, edges, k):
    """Recursive implementation of deletion-contraction algorithm"""
    # Base case: Empty graph
    if not edges:
        return k ** len(vertices)
    
    # Base case: Disconnected graph
    if not is_connected(vertices, edges):
        # Find connected components
        components = find_connected_components(vertices, edges)
        
        print(f"Graph is disconnected with {len(components)} components")
        # Compute polynomial for each component and multiply them
        result = 1
        for component_vertices, component_edges in components:
            comp_graph = build_graph(component_edges)
            poly = _chromatic_polynomial_recursive(comp_graph, component_vertices, component_edges, k)
            result *= poly
            print(f"Component polynomial: {poly}")
        
        return result
    
    # Choose an edge for deletion-contraction
    edge = edges[0]
    u, v = edge
    
    print(f"Applying deletion-contraction on edge {edge}")
    
    # Deletion: G - e
    deletion_edges = [e for e in edges if e != edge and e != (v, u)]
    deletion_graph = build_graph(deletion_edges)
    deletion_vertices = vertices.copy()
    
    print(f"  Deletion: removed edge {edge}")
    deletion_poly = _chromatic_polynomial_recursive(deletion_graph, deletion_vertices, deletion_edges, k)
    print(f"  Deletion polynomial: {deletion_poly}")
    
    # Contraction: G/e
    contraction_edges = []
    new_vertex = f"{u}-{v}"  # Merged vertex name
    
    # Update edges for the contracted graph
    for a, b in edges:
        if a == u and b == v or a == v and b == u:
            continue  # Skip the contracted edge
        
        # Replace occurrences of u or v with the new merged vertex
        new_a = new_vertex if a == u or a == v else a
        new_b = new_vertex if b == u or b == v else b
        
        if new_a != new_b:  # Avoid self-loops
            # Ensure the edge is not already present
            if (new_a, new_b) not in contraction_edges and (new_b, new_a) not in contraction_edges:
                contraction_edges.append((new_a, new_b))
    
    # Update vertices
    contraction_vertices = {new_vertex if v_i == u or v_i == v else v_i for v_i in vertices}
    
    print(f"  Contraction: merged vertices {u} and {v} into {new_vertex}")
    contraction_graph = build_graph(contraction_edges)
    contraction_poly = _chromatic_polynomial_recursive(contraction_graph, contraction_vertices, contraction_edges, k)
    print(f"  Contraction polynomial: {contraction_poly}")
    
    # Apply the recurrence relation: P(G) = P(G-e) - P(G/e)
    result = deletion_poly - contraction_poly
    print(f"  Result using deletion-contraction: {result}")
    
    return result

def build_graph(edges):
    """Build adjacency list from edges"""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def is_connected(vertices, edges):
    """Check if the graph is connected"""
    if not vertices:
        return True
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # DFS to check connectivity
    start_vertex = next(iter(vertices))
    visited = set()
    
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start_vertex)
    return len(visited) == len(vertices)

def find_connected_components(vertices, edges):
    """Find all connected components of the graph"""
    if not vertices:
        return []
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Find components using DFS
    components = []
    unvisited = vertices.copy()
    
    while unvisited:
        start = next(iter(unvisited))
        component_vertices = set()
        component_edges = []
        
        # DFS to find a component
        stack = [start]
        while stack:
            node = stack.pop()
            if node in component_vertices:
                continue
            
            component_vertices.add(node)
            unvisited.discard(node)
            
            for neighbor in graph[node]:
                if neighbor not in component_vertices:
                    stack.append(neighbor)
                    # Add this edge to component edges
                    if (node, neighbor) not in component_edges and (neighbor, node) not in component_edges:
                        component_edges.append((node, neighbor))
        
        components.append((component_vertices, component_edges))
    
    return components

def factor_polynomial(poly, k):
    """
    Factor the polynomial into a product of first-degree factors
    """
    expanded = sp.expand(poly)
    print(f"Expanded polynomial: {expanded}")
    
    factors = []
    degree = sp.degree(expanded, k)
    
    for i in range(degree):
        factors.append(k - i)
    
    # Verify the factorization
    product = 1
    for factor in factors:
        product *= factor
    
    assert sp.expand(product) == expanded, "Factorization failed"
    return factors

def format_output(poly, k):
    """Format the polynomial as a product of first-degree factors"""
    factors = factor_polynomial(poly, k)
    result = " * ".join([str(factor) for factor in factors])
    return result

def main():
    # Example usage
    edges = [(1, 2), (1, 3), (2, 3),(2,5),(3,4),(3,5),(4,5)]  # A triangle graph
    
    print("\nCalculating chromatic polynomial...")
    k = sp.Symbol('k')
    poly = chromatic_polynomial(edges)
    
    print("\nChromatic polynomial in expanded form:")
    print(sp.expand(poly))
    
    print("\nChromatic polynomial as product of first-degree factors:")
    print(format_output(poly, k))

if __name__ == "__main__":
    main()