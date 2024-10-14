# Tuple Operations

# 1. Creating a tuple
my_tuple = (10, 20, 30, 40)
print("Original tuple:", my_tuple)

# 2. Access elements
first_element = my_tuple[0]
print("First element:", first_element)

# 3. Index: Return the index of the first occurrence of a specific element
index_of_30 = my_tuple.index(30)
print("Index of 30:", index_of_30)

# 4. Count: Return the number of occurrences of a specific element
count_of_20 = my_tuple.count(20)
print("Count of 20:", count_of_20)

# 5. Concatenation: Join two tuples
tuple2 = (50, 60)
concatenated_tuple = my_tuple + tuple2
print("After concatenation:", concatenated_tuple)

# 6. Slicing: Access a part of the tuple
sliced_tuple = my_tuple[1:3]  # Elements from index 1 to 2
print("Sliced tuple:", sliced_tuple)

# 7. Unpacking: Assign tuple elements to variables
a, b, c, d = my_tuple
print("Unpacked values:", a, b, c, d)

# 8. Converting to a list (for operations like append)
my_list = list(my_tuple)
my_list.append(50)  # Append after converting to a list
my_tuple = tuple(my_list)  # Convert back to a tuple
print("After appending 50:", my_tuple)

# 9. Iterate through tuple
for item in my_tuple:
    print("Tuple element:", item)
