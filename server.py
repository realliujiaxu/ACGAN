# create an INET, STREAMing socket
import json
import socket
import threading
import base64
from io import BytesIO

from PIL.Image import Image
import PIL

from stylegan2_generator import Stylegan2Generator
import torch

class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, gan, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        self.gan = gan
        self.sample()

    def sample(self):
        content, noi = self.gan.sample(1, 'w')
        attribute = torch.ones([1, 40]) * 0.5

        self.content = content.cuda()
        self.noi = noi.cuda()
        self.attribute = attribute.cuda()

    def run(self):
        print("开启线程.....")
        try:
            while True:
                # 接受Springboot传来的指令数据
                msg = ''
                while True:
                    # 读取recvsize个字节
                    rec = self._socket.recv(self._recvsize)
                    # 解码
                    msg += rec.decode(self._encoding)
                    # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                    # 所以需要自定义协议标志数据接受完毕
                    if msg.strip().endswith('over'):
                        break

                print(msg)
                movement, attribute = msg[:-4].split(",")
                attribute = int(attribute)
                print(f"收到请求: movement {movement} on attribute {attribute}")
                # img = Image.fromarray(cv2.cvtColor(facetest(), cv2.COLOR_BGR2RGB))
                # img = PIL.Image.open("F:\\Work\\lanqiao\\GANDemo\\src\\main\\resources\\static\\images\\0.jpg")
                if movement == "increment":
                    self.attribute[:, attribute] += 0.1
                elif movement == "decrement":
                    self.attribute[:, attribute] -= 0.1
                else:
                    self.sample()

                imgs = self.gan.synthesize(torch.cat([self.content, self.attribute], dim=1), self.noi, 'w')
                img = self.gan.postprocess(imgs.cpu().numpy())[0]
                img = PIL.Image.fromarray(img)
                #创建一个BytesIO()
                output_buffer = BytesIO()
                #写入output_buffer
                img.save(output_buffer,format='JPEG')
                #在内存中读取
                byte_data = output_buffer.getvalue()
                #转化为base64
                sendmsg = base64.b64encode(byte_data)
                self._socket.sendall(str(sendmsg)[2:-1].encode(self._encoding))
                self._socket.sendall('^\n'.encode(self._encoding))
                print("处理完成")

        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
        finally:
            self._socket.close()
        print("任务结束.....")

    def __del__(self):
        pass

stylegan2 = Stylegan2Generator()

# content, noi = stylegan2.sample(1, 'w')
# attribute = torch.ones([1, 40]) * 0.5
# content = content.cuda()
# noi = noi.cuda()
# attribute = attribute.cuda()
# img = stylegan2.synthesize(torch.cat([content, attribute], dim=1), noi, 'w')

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bind the socket to a public host, and a well-known port
serversocket.bind(('', 6007))
# become a server socket
serversocket.listen(5)
while True:
    # accept connections from outside
    (clientsocket, address) = serversocket.accept()
    # now do something with the clientsocket
    # in this case, we'll pretend this is a threaded server
    ct = ServerThreading(stylegan2, clientsocket)
    ct.start()