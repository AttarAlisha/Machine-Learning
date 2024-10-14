# List Operations in Python

# Initial list
my_list = [10, 20, 30, 40]

# 1. Append: Add an element to the end of the list
my_list.append(50)
print("After append:", my_list)

# 2. Extend: Add elements from another list to the end of the current list
my_list.extend([60, 70])
print("After extend:", my_list)

# 3. Insert: Insert an element at a specific position
my_list.insert(2, 25)  # Insert 25 at index 2
print("After insert:", my_list)

# 4. Remove: Remove the first occurrence of a specific element
my_list.remove(40)  # Remove element 40
print("After remove:", my_list)

# 5. Pop: Remove and return the element at the given index (default is the last element)
removed_element = my_list.pop()  # Remove the last element
print("After pop:", my_list)
print("Popped element:", removed_element)

# 6. Clear: Remove all elements from the list
my_list.clear()
print("After clear:", my_list)

# 7. Index: Return the index of the first occurrence of a specific element
my_list = [10, 20, 30, 40, 50]  # Resetting list
index_of_30 = my_list.index(30)
print("Index of 30:", index_of_30)

# 8. Count: Return the number of occurrences of a specific element
count_of_20 = my_list.count(20)
print("Count of 20:", count_of_20)

# 9. Sort: Sort the elements of the list in ascending order
my_list.sort()
print("After sort:", my_list)

# 10. Reverse: Reverse the elements of the list
my_list.reverse()
print("After reverse:", my_list)

# 11. Split: Convert a string into a list by splitting it at a specified delimiter
my_string = "apple,banana,cherry"
split_list = my_string.split(",")
print("After split:", split_list)
