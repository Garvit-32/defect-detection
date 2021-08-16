from glob import glob
import pandas as pd
from multiprocessing import Pool
import os
import cv2
import shutil
from tqdm import tqdm


def process(i, output_folder):

    try:
        if type(i) == str:
            f = open(
                os.path.join(
                    output_folder,
                    "labels",
                    i[:-4] + ".txt",
                ),
                "w+",
            )

            shutil.copy(os.path.join(
                "Images",
                i
            ), os.path.join(
                output_folder,
                "images",
                i,
            ))
        else:

            f = open(
                os.path.join(
                    output_folder,
                    "labels",
                    i[0][:-4] + ".txt",
                ),
                "w+",
            )

            f.write(
                "{} {} {} {} {}\n".format(
                    0,  i[1], i[2], i[3], i[4]
                )
            )
            shutil.copy(os.path.join(
                "Images",
                i[0]
            ), os.path.join(
                output_folder,
                "images",
                i[0],
            ))

    except FileNotFoundError:
        pass


if __name__ == "__main__":
    # open txt file
    # convert to yolo

    # write new txt file

    # fcos_txt_path = "/home/sanchit/Downloads/Compressed/single_train_plates.txt"
    output_folder = "data1/defect"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

    df = pd.read_csv('Train_DefectBoxes_PrithviAI.csv')

    # print(df.head())

    # fcos_f = open(
    #     fcos_txt_path,
    #     "r",
    # )
    task_list = []

    main_folder = os.listdir('Images')
    # print(val)

#    for i in tqdm(df.values):

    for i in tqdm(main_folder):
        val = df.loc[df['image_id'] == i].values
        # if len(val) == 0:
        #     task_list.append((i, output_folder))
        if len(val) != 0:
            task_list.append((val[0], output_folder))
        # else:
        #     task_list.append((val[0], output_folder))

   # process(i, output_folder)

    pool = Pool(4)  # number of workers
    pool.starmap(process, task_list, chunksize=1)
    pool.close()
    pool.join()
   # python train.py --project defect --batch-size 8 --img 1000 --data data/defect.yaml --weights yolov5x.pt --hyp data/hyp.plate.yaml --epochs 100
    # python train.py --img 640 --batch 2 --epochs 3 --data data/defect.yaml --weights yolov5m.pt --project defect
