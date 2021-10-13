import os
from os import path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_batch_real(root_path, object=None):
    # root path: to the YCBSight-Real
    if object is None:
        # visualize all objects' data
        object_folders = sorted(os.listdir(root_path))
    else:
        object_folders = [object]

    for obj in object_folders:
        if obj == ".DS_Store":
            continue
        print("Visualizing data for " + obj + " ...")
        current_path = osp.join(root_path, obj)

        # only tactile images
        tactile_path = osp.join(current_path, "gelsight")
        tactiles = sorted(os.listdir(tactile_path), key=lambda y: int(y.split("_")[1]))

        for i, tactile in enumerate(tactiles):
            if i % 2 == 0:
                # skip all background images
                continue
            cur_tactile = osp.join(tactile_path, tactile)
            tactile_img = cv2.imread(cur_tactile)
            cv2.imshow("tactile image # " + str(i), tactile_img)
            cv2.waitKey(0)


def visualize_batch_sim(root_path, object=None):
    # root path: to the YCBSight-Sim
    if object is None:
        # visualize all objects' data
        object_folders = sorted(os.listdir(root_path))
    else:
        object_folders = [object]

    for obj in object_folders:
        if obj == ".DS_Store":
            continue
        print("Visualizing data for " + obj + " ...")
        current_path = osp.join(root_path, obj)

        # tactile images, height maps, contack masks
        tactile_path = osp.join(current_path, "gelsight")
        height_path = osp.join(current_path, "gt_height_map")
        contact_path = osp.join(current_path, "gt_contact_mask")

        tactiles = sorted(os.listdir(tactile_path), key=lambda y: int(y.split(".")[0]))
        heightMaps = sorted(os.listdir(height_path), key=lambda y: int(y.split(".")[0]))
        contactMasks = sorted(os.listdir(contact_path), key=lambda y: int(y.split(".")[0]))

        for i, heightMap in enumerate(heightMaps):
            cur_heightMap = osp.join(height_path, heightMaps[i])
            cur_tactile = osp.join(tactile_path, tactiles[i])
            cur_contactMask = osp.join(contact_path, contactMasks[i])

            height_map = np.load(cur_heightMap)
            tactile_img = cv2.imread(cur_tactile)
            tactile_img = cv2.cvtColor(tactile_img, cv2.COLOR_RGB2BGR)
            contact_mask = np.load(cur_contactMask)

            fig = plt.figure(figsize=(8, 4))
            fig.suptitle("Tactile image/height map/contact mask # " + str(i), fontsize=16)
            ax = plt.subplot("131")
            ax.imshow(tactile_img/255.0)
            ax.axis('off')

            ax = plt.subplot("132")
            ax.imshow(height_map)
            ax.axis('off')

            ax = plt.subplot("133")
            ax.imshow(contact_mask)
            ax.axis('off')
            plt.show()


if __name__ == "__main__":

    root_path = osp.join("..", "..")
    sim = True # True
    obj = '002_master_chef_can' # '002_master_chef_can'
    if sim:
        sim_path = osp.join(root_path, "YCBSight-Sim")
        visualize_batch_sim(sim_path, object=obj)
    else:
        real_path = osp.join(root_path, "YCBSight-Real")
        visualize_batch_real(real_path, object=obj)
