# Set Operations

# 1. Creating a set
my_set = {1, 2, 3}
print("Original set:", my_set)

# 2. Adding an element to the set
my_set.add(4)
print("After adding 4:", my_set)

# 3. Removing an element using remove()
my_set.remove(2)
print("After removing 2:", my_set)

# 4. Discard: Removes an element if present, does nothing if not found
my_set.discard(5)  # No error if 5 is not present
print("After discard 5:", my_set)

# 5. Pop: Removes and returns a random element from the set
popped_element = my_set.pop()
print("After pop:", my_set)
print("Popped element:", popped_element)

# 6. Union: Return a new set with elements from both sets
set2 = {5, 6, 7}
union_set = my_set.union(set2)
print("Union of sets:", union_set)

# 7. Intersection: Return a new set with elements common to both sets
set3 = {3, 4, 7}
intersection_set = my_set.intersection(set3)
print("Intersection of sets:", intersection_set)

# 8. Difference: Return a new set with elements in the first set but not the second
difference_set = my_set.difference(set3)
print("Difference of sets:", difference_set)

# 9. Clearing the set
my_set.clear()
print("After clearing the set:", my_set)
