import os
from random import randint
from typing import Set

def no_mask_divider():
    no_mask_path = "./dataset/Train/WithoutMask/"
    filelist = []
    for (dirpath, dirnames, filenames) in os.walk(no_mask_path):
        filelist.extend(filenames)
        break
    numbers = []

    for _ in range(350):
        value = randint(0, len(filelist))
        numbers.append(value)

    final_filelist = set()

    for number in numbers:
        final_filelist.add(filelist[number])
    

    for file in final_filelist:
        os.rename(no_mask_path+file, "./no_mask/"+file)
    
def mask_divider():

    names = ["tanzia","anagh","nadib"]
    for name in names:
        mask_path = "./dataset/Train/WithMask/"
        filelist = []
        for (dirpath, dirnames, filenames) in os.walk(mask_path):
            filelist.extend(filenames)
            break
        numbers = []
        for _ in range(350):
            value = randint(0, len(filelist))
            numbers.append(value)

        final_filelist = set()

        for number in numbers:
            final_filelist.add(filelist[number])
        

        for file in final_filelist:
            os.rename(mask_path+file, "./"+name+"/"+file)
           

if __name__ == "__main__":
    mask_divider()
