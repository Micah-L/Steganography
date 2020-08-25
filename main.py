
import sys
import cv2 as cv
import numpy as np
from itertools import cycle
import ctypes

from steg import *



MODE = "PREDEFS"
#MODE = "INPUT"
CARRIER_IMAGE_NAME = "4.jpg"
INNER_IMAGE_NAME = "tfs8.34.png" #"hideable_text.png"
B = 2
READONLY = False
CHECK_STRIDE = False #check max_stride only then exit
WRITE_MODES = [""]

if __name__ == "__main__":
#    pass
#else:
    if len(sys.argv) >= 3:
        carrier_image_name = sys.argv[1]
        inner_image_name = sys.argv[2]
        if len(sys.argv) >= 4:
            b = int(sys.argv[4])
        else:
            b = B
    else:
        carrier_image_name = input("Carrier image name: ") if MODE == "INPUT" else CARRIER_IMAGE_NAME
        inner_image_name = input("Inner image name: ") if MODE == "INPUT" else INNER_IMAGE_NAME
        b = B

    img = Image()
    img.load_from_filename(carrier_image_name)

    with open(inner_image_name, 'rb') as f:
        inner_img = f.read()

    msg_length = len(inner_img)
    print(f"The message file contains {msg_length} bytes.")
    print(f"Maximum allowable stride: {b * np.prod( img.img_arr.shape ) // ( 8 * len(inner_img) )}")
    print(f"Maximum bytes writeable with a stride of 1: {b * np.prod( img.img_arr.shape )//8}")
    s = int(input("Stride: ")) if MODE == "INPUT" else 2

    seed = 9123129395 % 2**32
    np.random.seed(seed)
    #s = cycle( iter( np.random.randint(10,30,size= 1000000)  ) )

    if CHECK_STRIDE:
        exit()

    img_holding_message_name = "image_holding_message"
    recovered_inner_image_name = "recovered_inner_image"

    saved_name = f'{img_holding_message_name}.png'
    if not READONLY:
        print("Writing inner image... ")
        img.write_bytes(bytes_to_write = inner_img, num_writeable_bits = b, stride = s)
        img.save_as(saved_name)
        print(f"Saved as {saved_name}")

    img.load_from_filename(saved_name)

    print("Reading inner image... ")
    new_inner_img = img.read_bytes(num_bytes_to_read = msg_length, num_writeable_bits= b, stride = s)
    saved_name = f'{recovered_inner_image_name}.png'
    with open(saved_name, 'wb') as f:
        f.write(new_inner_img)
    print(f"Saved as {saved_name}")