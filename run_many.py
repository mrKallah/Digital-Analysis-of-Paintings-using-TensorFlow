import CNN as cnn
from datetime import timedelta
import time
#   print_many(img_size, num_chan, num_aug, fs1, num_fs1, fs2, num_fs2, size_fc, num_optim_iter):

start_time = time.time()


cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 100)

cnn.run_many(100, 1, 10, 10, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 20, 10, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 30, 10, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 40, 10, 16, 10, 36, 128, 100)

cnn.run_many(100, 1, 1, 4, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 8, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 24, 16, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 32, 16, 10, 36, 128, 100)

cnn.run_many(100, 1, 1, 10, 5, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 15, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 20, 10, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 30, 10, 36, 128, 100)

cnn.run_many(100, 1, 1, 10, 16, 5, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 15, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 20, 36, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 30, 36, 128, 100)

cnn.run_many(100, 1, 1, 10, 16, 10, 16, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 24, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 48, 128, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 64, 128, 100)

cnn.run_many(100, 1, 1, 10, 16, 10, 36, 1, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 64, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 96, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 256, 100)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 512, 100)

cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 25)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 50)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 200)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 400)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 800)
cnn.run_many(100, 1, 1, 10, 16, 10, 36, 128, 1000)

cnn.run_many(100, 3, 1, 10, 16, 10, 36, 128, 100)

cnn.run_many(1, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(10, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(25, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(50, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(200, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(300, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(400, 1, 1, 10, 16, 10, 36, 128, 100)
cnn.run_many(500, 1, 1, 10, 16, 10, 36, 128, 100)


end_time = time.time()
time_dif = end_time - start_time
time_dif = str(timedelta(seconds=int(round(time_dif))))


print("Total Time = {}".format(time_dif))