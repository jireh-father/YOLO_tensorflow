import YOLO_small_tf
import os
import glob
import sys
import shutil

yolo = YOLO_small_tf.YOLO_TF()

batch_size = 256

image_path = "D:\data\main_image\image\processing\dabainsang\dataset\main_image"

not_person_path = "D:\data\main_image\image\processing\dabainsang\dataset\main_image_not_person"
if not os.path.isdir(not_person_path):
    os.makedirs(not_person_path)

image_files = glob.glob(os.path.join(image_path, "*"))
image_cnt = len(image_files)
if image_cnt < 1:
    sys.exit()

for i in range(0, image_cnt, batch_size):
    batch_image_files = image_files[i:i + batch_size]
    results = yolo.is_person_from_files(batch_image_files)
    for i, result in enumerate(results):
        if not result:
            shutil.move(batch_image_files[i], os.path.join(not_person_path, os.path.basename(batch_image_files[i])))
            print("%s : copied to not person dir" % os.path.basename(batch_image_files[i]))
