import os
def CrawlKittiDataPath(base_dir, sequences, camera):
    """get all image_paths under base_dir
    """
    image_paths = []
    for sequence in sequences:
        image_dir = os.path.join(base_dir, sequence, camera)
        image_rel_list = os.listdir(image_dir)
        image_abs_list = [os.path.join(image_dir, image_rel) for image_rel in image_rel_list]
        image_paths += image_abs_list
        print(image_paths[-1])
    return image_paths
    
if __name__ == "__main__":
    base_dir = "#####/KittiOdom/sequences/"
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    image_paths = CrawlKittiDataPath(base_dir, sequences, "image_3")
    # print(image_paths)