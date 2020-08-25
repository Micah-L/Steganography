
import sys
import cv2 as cv
import numpy as np

from itertools import cycle
import ctypes


def get_max_stride(carrier_arr, msg_arr, bits = 1):
    return bits * np.prod( carrier_arr.shape ) // ( 8 * np.prod(msg_arr.shape) )

def iterable_is_allowable(itr, carrier_arr, msg_arr, bits = 1):
    """ Check if the given iterable will fit the message image inside the carrier image when used as a sequence of skips and returns a bool.
        Returns true if sum_{i=1}^{bits_to_write} next(itr) <= writeable_bits
        Returns false if the iterable runs out or the sum is too large.
    """
    summation = 0
    num_bits_to_write = np.prod(msg_arr.shape)*8
    writeable_bits = np.prod(carrier_arr.shape)*bits
    for i, x in enumerate(itr):
        if summation > writeable_bits:
            return False
        elif i >= num_bits_to_write:
            return True
        summation += x

def get_max_inner_image_size(img_arr, bits, stride):
    pass

def get_itr(mat, step = (1,1,1) , randomize = None, stride = 1, seed = None):
    """ Works for size n x m x l matrices. Yields the next position i, j, k
        
        stride  - The number of iterations skip forward.
                  Can be constant or an iterable.
                  A stride of 1 does not skip any iterations. 
        seed    - If seed is not None, stride should be a 2-tuple, and the stride is randomized each time with the seed.  """
    iteration_number = -1
    if seed is not None:
        s = np.random.default_rng(stride[0], stride[1], seed=seed)
    try:
        # Allow stride to be a generator/sequence
        for i in range(0, len(mat), step[0]):
            for j in range(0, len(mat[i]), step[1]):
                for k in range(0, len(mat[i][j]), step[2]):
                    s = next(stride)
                    iteration_number = (iteration_number + 1) % s
                    if iteration_number == 0:
                        yield i, j, k
    except TypeError:
        # Should be an integer then
        for i in range(0, len(mat), step[0]):
            for j in range(0, len(mat[i]), step[1]):
                for k in range(0, len(mat[i][j]), step[2]):
                    iteration_number = (iteration_number + 1) % stride
                    if iteration_number == 0:
                        yield i, j, k

    
def num_to_binary_array(num, min_length = 8):
    arr = []
    while num > 0:
        arr.append(num % 2)
        num >>= 1
    while len(arr) < min_length:
        arr.append(0)
    arr.reverse()
    return arr

def binary_array_to_num(arr):
    arr.reverse()
    total = 0
    for i in range(len(arr)):
        total = total + (2**i)*arr[i]
    return total

def str_to_bits(s):
    byte_arr = np.array([ord(x) for x in s], dtype=np.uint8)
    for b in byte_arr:
        bits = bin(b)
        next_bits = [0,0,0,0,0,0,0,0]
        i = 1
        while bits[-1*i] != 'b':
            next_bits[-1*i] = bits[-1*i]
            i += 1
        for nb in next_bits:
            yield nb

def build_str_from_bits(curr_str,curr_bits,curr_bit):
    curr_bits.append(curr_bit)
    if len(curr_bits) == 8:
        char = 0
        for i in range(8):
            char += curr_bits[i] * 2**(7-i)
        curr_str += chr(char)
        curr_bits = []
    return curr_str, curr_bits

class Image:
    def __init__(self, img_arr = None):
        self.img_arr = img_arr
    def load_from_filename(self,filename):
        self.img_arr = cv.imread(filename)
    def save_as(self, filename):
        cv.imwrite(filename, self.img_arr)
    def write_bit(self, ijk, bit, place = 1):
        """ Deprecated. Use write_bits. """
        i, j, k = ijk
        self.img_arr[i][j][k] =  ( (int(self.img_arr[i][j][k]) >> place ) << place ) | (int(bit)*2**(place-1)) | (self.img_arr[i][j][k] & (2**place-1) )
    def read_least_bit(self, ijk): 
        i,j,k = ijk
        return self.img_arr[i][j][k] & 1
    def write_bits(self, ijk, bits, num_bits = None, place = 1):
        """ Writes all the bits of bits into the i,j,k position of the image array.
            The lsb of bits will always end up in place (with place=1 being the lsb). """
        i,j,k = ijk
        if num_bits is None:
            num_bits = len(bin(bits)) - 2
        self.img_arr[i][j][k] = self.img_arr[i][j][k] >> place + num_bits - 1 << place + num_bits - 1 | bits * 2**(place - 1) | self.img_arr[i][j][k] & 2**(place - 1) - 1

    def read_bits(self, ijk, highest_place, lowest_place=1):
        """ Reads the bits  from the i,j,k position of the image.
            Selects the bits between highest_place and lowest_place, where place
            is an integer from 1 to 8, with 1 being the lsb and 8 being the msb.
            e.g. img[i][j][k] = 01101100
                                 h    l 
            highest_place = 7
            lowest_place = 2
            then return 110110 as a number"""
        i,j,k = ijk
        f = sum([2**i for i in range(lowest_place - 1, highest_place)])
        return (f & self.img_arr[i][j][k]) >> (lowest_place - 1)

    def write_bytes(self, bytes_to_write, num_writeable_bits = 2, stride = 1):
        """ bytes_to_write is an array of bytes to be hidden in the image.
            bits of bytes_to_write are written in the least num_writeable_bits significant bits of the image.
            stride is allowed to be an integer of an interable and indicates how many bytes to move between each write. Stride should not be 0 or contain 0s. 
        """
        itr_carrier = get_itr(self.img_arr, stride=stride)
        this_write_bit = num_writeable_bits
        for this_byte in bytes_to_write:
            this_byte = num_to_binary_array(this_byte)
            for this_bit in this_byte:
                try:
                    i, j, k = next(itr_carrier)
                    self.write_bits( (i, j, k), this_bit, num_bits = 1, place = this_write_bit)
                except StopIteration:
                     # Allow for bits-1 number of StopIterations before we're really out of space
                    if this_write_bit == 1: #if we were already writing in the lsb, we're out of space
                        raise Exception("ran out of space to write the image")
                    else:
                        this_write_bit  -= 1
                        itr_carrier = get_itr(self.img_arr, stride=stride)
                        i, j, k = next(itr_carrier)
                        self.write_bits( (i, j, k), this_bit, num_bits = 1, place = this_write_bit)

    def read_bytes(self, num_bytes_to_read, num_writeable_bits = 2, stride = 1):
        hidden_bytes = []
        itr_carrier = get_itr(self.img_arr, stride = stride)
        this_read_bit = num_writeable_bits
        for _ in range(num_bytes_to_read):
            this_byte = []
            for t in range(8):
                try:
                    i,j,k = next(itr_carrier)
                    this_bit = self.read_bits( (i,j,k), highest_place = this_read_bit, lowest_place = this_read_bit)
                    this_byte.append(this_bit)
                except StopIteration:
                    #Allow for bits-1 number of StopIterations before we can't read anymore.
                    if this_read_bit == 1: 
                        raise Exception("ran out of space to read the image")
                    else:
                        this_read_bit -= 1
                        itr_carrier = get_itr(self.img_arr, stride = stride)
                        i,j,k = next(itr_carrier)
                        this_bit = self.read_bits( (i,j,k), highest_place = this_read_bit, lowest_place = this_read_bit)
                        this_byte.append(this_bit)
            hidden_bytes.append( binary_array_to_num( this_byte ) )

        return bytes(hidden_bytes)


    ##### The following methods are deprecated. Use write_bytes and read_bytes for general handling of all file embedding #####

    def write_inner_img_sequential(self, inner_img, bits = 1, stride = 1):
        """ 
            bits - The number of bits to use in the carrier image
            stride - The number of positions in the carrier image to move at a time. If stride > 1, positions inbetween are skipped instread of written to. Skipping more positions may decrease the distortion, but leaves less space available.
        """
        itr_inner = get_itr(inner_img)
        itr_carrier = get_itr(self.img_arr, stride = stride)
        this_write_bit = bits # start writing in the most significant bit available
        for i, j, k in itr_inner:
            this_byte = num_to_binary_array( inner_img[i][j][k] )
            # chunks = [ binary_array_to_num( this_byte[t:t+bits] ) for t in range(0,8,bits)]
            for this_bit in this_byte:
                try:
                    ii, jj, kk = next(itr_carrier)
                    self.write_bits( (ii, jj, kk), this_bit, num_bits = 1, place=this_write_bit )
                except StopIteration:
                    # Allow for bits-1 number of StopIterations before we're really out of space
                    if this_write_bit == 1: #if we were already writing in the lsb, we're out of space
                        raise Exception("ran out of space to write the image")
                    else:
                        this_write_bit  -= 1
                        print(f"Reducing write bit to {this_write_bit}")
                        itr_carrier = get_itr(self.img_arr, stride = stride)
                        ii, jj, kk = next(itr_carrier)
                        self.write_bits( (ii, jj, kk), this_bit, num_bits = 1, place=this_write_bit )

                
    def read_inner_img_sequential(self, shape, bits = 1, stride = 1):
        inner_img = np.zeros(shape=shape, dtype=np.uint8)
        itr_inner = get_itr(inner_img)
        itr_carrier = get_itr(self.img_arr, stride = stride)    
        this_read_bit = bits # start reading from the most siginicant bit availabale 
        for i, j, k in itr_inner:
            this_byte = []
            for t in range(0,8):
                try:
                    ii, jj, kk = next(itr_carrier)
                    this_bit = self.read_bits( (ii,jj,kk), highest_place = this_read_bit, lowest_place = this_read_bit)
                    this_byte.append(this_bit)
                except StopIteration:
                    #Allow for bits-1 number of StopIterations before we can't read anymore.
                    if this_read_bit == 1: 
                        raise Exception("ran out of space to read the image")
                    else:
                        print(f"Now reading from bit {this_read_bit}")
                        this_read_bit -= 1
                        itr_carrier = get_itr(self.img_arr, stride = stride)
                        ii,jj,kk=next(itr_carrier)
                        this_bit = self.read_bits( (ii,jj,kk), highest_place = this_read_bit, lowest_place = this_read_bit)
                        this_byte.append(this_bit)
            inner_img[i][j][k] = binary_array_to_num( this_byte )

        return inner_img
            

    def write_inner_img_1(self, inner_img, write_bit = 1):
        step = (8,1,1)
        itr = get_itr(self.img_arr, step = step)
        for i,j,k in itr:
            this_byte = num_to_binary_array( inner_img[ i // step[0] % inner_img.shape[0], j // step[1] % inner_img.shape[1], k // step[2] % inner_img.shape[2]] )
            for t in range(8):
                if i+t < self.img_arr.shape[0]:
                    self.write_bits((i+t, j, k), bits = this_byte[t], place=write_bit)

    def read_inner_img_1(self, write_bit = 1):
        inner_img = np.zeros( (self.img_arr.shape[0]//8, self.img_arr.shape[1], self.img_arr.shape[2]))
        itr = get_itr(inner_img)
        for i, j, k in itr:
            try:
                this_byte =  [ self.read_bits( (8*i + t, j, k), write_bit, write_bit) for t in range(8) ]
                this_byte = binary_array_to_num( this_byte )
            except IndexError:
                this_byte = 0
            inner_img[i][j][k] = this_byte
        return inner_img

    def write_inner_img_2(self, inner_img, step = (4,1,1)):
        itr = get_itr(self.img_arr, step = step)
        for i,j,k in itr:
            this_byte = num_to_binary_array( inner_img[ i // step[0] % inner_img.shape[0], j // step[1] % inner_img.shape[1], k // step[2] % inner_img.shape[2]] )
            for t in range(4):
                if i+t < self.img_arr.shape[0]:
                    self.write_bits((i+t, j, k), bits = this_byte[t], place=4)
                    self.write_bits((i+t, j, k), bits = this_byte[4+t], place=3)

    def read_inner_img_2(self):
        inner_img = np.zeros( (self.img_arr.shape[0]//4, self.img_arr.shape[1], self.img_arr.shape[2]))
        itr = get_itr(inner_img)
        for i, j, k in itr:
            try:
                this_byte =  [ self.read_bits( (4*i + t, j, k), 4, 4) for t in range(4) ]
                this_byte += [ self.read_bits( (4*i + t, j, k), 3, 3) for t in range(4) ]
                this_byte = binary_array_to_num( this_byte )
            except IndexError:
                this_byte = 0
            inner_img[i][j][k] = this_byte
        return inner_img
    def write_inner_img(self, inner_img, bits_used = (1)):
        """ Supported tuples for bits_used are any singleton (x),
            and any 2-tuple or any 4-tuple (4,3,2,1) in descending order """
        step_size = 8 // len(bits_used)
        itr = get_itr(self.img_arr, step = (step_size,1,1))
        for i, j, k in itr:
            # Byte from inner image to write
            this_byte = inner_img[ i//step_size % inner_img.shape[0] ][ j % inner_img.shape[1] ][ k % inner_img.shape[2] ]
            this_byte = num_to_binary_array(this_byte)
            for t in range(step_size):
                if i+t < self.img_arr.shape[0]:
                    self.write_bits( (i + t, j, k), this_byte[t])
    def read_inner_img(self, bits_used = 1):
        inner_img = np.zeros( (self.img_arr.shape[0]//8, self.img_arr.shape[1], self.img_arr.shape[2]), dtype=np.uint8 )
        itr = get_itr(inner_img)
        for i, j, k in itr:
            try:
                this_byte = [self.read_least_bit((8*i+t,j,k)) for t in range(8)] # [self.img_arr[8*i+t][j][k] & 1 for t in range(8)]
                this_byte = binary_array_to_num(this_byte)
            except IndexError:
                this_byte = 0
            
            inner_img[i][j][k] = this_byte
        return inner_img
    def write_inner_string(self, s):
        itr = get_itr(self.img_arr)
        s = cycle(str_to_bits(s)) # repeat the string to fill the image
        for i,j,k in itr:
            b = next(s)
            self.write_bit( (i,j,k), b)
    def read_inner_string(self):
        itr = get_itr(self.img_arr)
        s = ""
        bits = []
        for i,j,k in itr:
            curr_bit = int(img[i][j][k]) & 1
            s, bits = build_str_from_bits(s, bits, curr_bit)