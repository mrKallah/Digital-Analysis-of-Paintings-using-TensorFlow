import Train as cnn
from datetime import timedelta
import time





# Starts a timer to see the total runtime of all the augmentations
start_time = time.time()

# Cause xming crash if xming is not running bur required
cnn.crash_xming()


cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 25, False)
cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 50, False)
cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 200, False)
cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 400, False)
cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 800, False)
cnn.run_many(50, 3, 40, 16, 16, 10, 36, 128, 1000, False)




# Calculates how long the program has used to run
end_time = time.time()
time_dif = end_time - start_time

# Reformats the time-data to a easy to read format
time_dif = str(timedelta(seconds=int(round(time_dif))))

# Prints total time taken
print("Total Time = {}".format(time_dif))