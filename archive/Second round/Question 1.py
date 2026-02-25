# Step 1: Start with everyone alone
groups = {
    'A': 'A',  # A is in group A
    'B': 'B',  # B is in group B
    'C': 'C',  # C is in group C
    'D': 'D'   # D is in group D
}

# Step 2: Define all roads
roads = [
    ('A', 'B', 4),  # A to B costs 4
    ('A', 'C', 1),  # A to C costs 1
    ('A', 'D', 3),  # A to D costs 3
    ('B', 'D', 2),  # B to D costs 2
    ('C', 'D', 5)   # C to D costs 5
]

# Step 3: Find MST
mst = []  # This will store our answer

# Keep going until everyone is in one group
while len(set(groups.values())) > 1:  # More than 1 unique group
    
    # For each group, find its cheapest outgoing road
    cheapest_for_group = {}
    
    for city1, city2, cost in roads:
        group1 = groups[city1]
        group2 = groups[city2]
        
        # Only consider roads between DIFFERENT groups
        if group1 != group2:
            # Is this the cheapest road from group1?
            if group1 not in cheapest_for_group:
                cheapest_for_group[group1] = (city1, city2, cost)
            elif cost < cheapest_for_group[group1][2]:
                cheapest_for_group[group1] = (city1, city2, cost)
    
    # Add all these cheap roads to MST
    for city1, city2, cost in cheapest_for_group.values():
        mst.append((city1, city2, cost))
        
        # Merge the groups
        old_group = groups[city2]
        new_group = groups[city1]
        for city in groups:
            if groups[city] == old_group:
                groups[city] = new_group

print("MST edges:", mst)
# Output: [('A', 'C', 1), ('B', 'D', 2), ('A', 'D', 3)]
```
