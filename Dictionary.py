# Dictionary Operations

# 1. Creating a dictionary
my_dict = {"a": 1, "b": 2, "c": 3}
print("Original dictionary:", my_dict)

# 2. Accessing elements by key
value_of_a = my_dict["a"]
print("Value of 'a':", value_of_a)

# 3. Adding a new key-value pair
my_dict["d"] = 4
print("After adding new key 'd':", my_dict)

# 4. Updating an existing key's value
my_dict["b"] = 5
print("After updating value of 'b':", my_dict)

# 5. Removing a key-value pair using pop()
removed_value = my_dict.pop("c")
print("After popping 'c':", my_dict)
print("Removed value of 'c':", removed_value)

# 6. Getting all keys
keys = my_dict.keys()
print("Keys in dictionary:", keys)

# 7. Getting all values
values = my_dict.values()
print("Values in dictionary:", values)

# 8. Getting all key-value pairs
items = my_dict.items()
print("Key-value pairs in dictionary:", items)

# 9. Checking if a key exists
if "a" in my_dict:
    print("'a' exists in the dictionary")

# 10. Clearing the dictionary
my_dict.clear()
print("After clearing the dictionary:", my_dict)
