import redis, struct, random, plyvel
import numpy as np
from typing import List

class Experience_Pool:
    def __init__(self):
        self.conn = None
        self.Len = 0
        self.__connect_exp_pool(path='./exp_pool')

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[np.ndarray]:
        type_dict = {
            1 : (1, np.uint8),
            2 : (2, np.float16),
            3 : (4, np.float32),
            4 : (8, np.float64),
        }
        index, length = 0, len(bytes_data)
        list_np_data = []
        while index < length:
            c_type = struct.unpack(">h", bytes_data[index: index+2])[0]
            tb, t = type_dict[c_type]
            count, shape_dim = struct.unpack(">II", bytes_data[index+2: index+10])
            shape = struct.unpack(">" + "I" * shape_dim,
                                  bytes_data[index + 10: index + 10 + shape_dim * 4])
            index = index + 10 + shape_dim * 4
            each_bytes_data = bytes_data[index: index + count*tb]
            list_np_data.append(np.frombuffer(each_bytes_data, dtype=t).reshape(*shape))
            index += count*tb
        return list_np_data

    @classmethod
    def __encode_data(cls, np_list_data: List[np.ndarray]) -> bytes:
        def type_encode(data_dtype: np.dtype):
            if data_dtype == np.uint8:
                return 1
            elif data_dtype == np.float16:
                return 2
            elif data_dtype == np.float32:
                return 3
            elif data_dtype == np.float64:
                return 4
            else:
                raise ValueError
        bytes_out = b""
        for np_data in np_list_data:
            shape = np_data.shape
            shape_dim = len(shape)
            count = 1
            for shape_i in shape:
                count *= shape_i
            bytes_out += struct.pack(">h", type_encode(np_data.dtype))
            bytes_out += struct.pack(">II", count, shape_dim)
            bytes_out += struct.pack(">" + "I" * shape_dim, *shape)
            bytes_out += np_data.tobytes()
        return bytes_out

    @classmethod
    def __data_to_exp(cls, data):
        exp = {}
        exp["states"] = []
        exp["reward"] = []
        i = 0
        while i < len(data):
            exp["states"].append([data[i], data[i+1], data[i+2]])
            i += 3
            if i >= len(data):
                break
            exp["reward"].append(data[i])
            i += 1
        exp["reward"].append(np.zeros_like(exp["reward"][-1]))
        length = len(exp["reward"])
        for j in range(2, length+1):
            exp["reward"][length-j] = np.add(exp["reward"][length-j], exp["reward"][length-j+1])
        return exp

    def __connect_exp_pool(self, path):
        self.conn = plyvel.DB(path, create_if_missing=False)
        self.Len = int(str(self.conn.get("Len".encode('UTF-8'), "0".encode('UTF-8')), encoding='UTF-8'))

    def random_get(self):
        return Experience_Pool.__data_to_exp(Experience_Pool.__decode_data(self.conn.get("{}".format(random.randint(0, self.Len-1)).encode('UTF-8'))))

    def close(self):
        if self.conn != None:
            self.conn.close()
            self.conn = None



# Ep = Experience_Pool()
# print(len(Ep.random_get("False")["reward"]))