import YOLO_face_tf
import os
import glob
import sys
import shutil

yolo = YOLO_face_tf.YOLO_TF()

batch_size = 30

image_path = "D:\data\main_image\image\processing\dabainsang\split\main_image\main_image_not_person"

not_person_path = "D:\data\main_image\image\processing\dabainsang\split\main_image\main_image_not_person_face"
if not os.path.isdir(not_person_path):
    os.makedirs(not_person_path)

image_files = glob.glob(os.path.join(image_path, "*"))
image_cnt = len(image_files)
if image_cnt < 1:
    sys.exit()

for i in range(0, image_cnt, batch_size):
    print(i, image_cnt)
    batch_image_files = image_files[i:i + batch_size]
    results = yolo.is_person_from_files(batch_image_files)
    for i, result in enumerate(results):
        if result:
            shutil.move(batch_image_files[i], os.path.join(not_person_path, os.path.basename(batch_image_files[i])))
            # print("%s : copied to not person dir" % os.path.basename(batch_image_files[i]))
