import array
import struct
import wave


file = open('../data/test5.wav', 'rb')
info = file.read()
print(info)
print(info[22:24])
print(struct.unpack('h', info[0:2])[0])
print(struct.unpack('<i', info[40:44])[0])
file = wave.open('demo1.wav', 'wb')
file.setnchannels(2)
file.setsampwidth(2)
file.setframerate(16000)
file.writeframes(info)
file.close()
print('write successfully')
# file_size = file.seek(0, 2)
# n = (file_size - 44) // 2
# # 生成buf
# buf = array.array('h', (0 for _ in range(n)))
# file.seek(44)
# # 将数据读入到buf中
#
#
# file.readinto(buf)
#
# # 写入数据
# f = open('demo.wav', 'wb')
# f.write(info)
# buf.tofile(f)
# # 关闭文件
# f.close()
# file.close()



