from datetime import datetime

# Get current date and time
now = datetime.now()
print(now)

# Create a datetime object
date_obj = datetime(2024, 9, 27, 14, 30)  # year, month, day, hour, minute
print(date_obj)

# Convert string to datetime
date_str = '2024-09-27 14:30:00'
date_format = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
print(date_format)

# Convert datetime to string
formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
print(formatted_date)

# Difference between dates
time_diff = datetime.now() - date_obj
print(time_diff)



