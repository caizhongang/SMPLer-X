import argparse
import socket
import itchat
import errno
import time
import threading

import os
import subprocess

parser = argparse.ArgumentParser(description="simple server!")
parser.add_argument('--mode', type=str)
parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=int)
parser.add_argument('--timeout', type=int)

class MsgServer(object):
    def __init__(self, server_ip, server_port):
        self.init_chat()
        self.send('server chat logged in!')
        self.start_server(server_ip, server_port)

    def init_chat(self):
        itchat.auto_login(enableCmdQR=2)

    def send(self, msg, echo=True):
        if echo:
            print(msg)
        itchat.send(msg, toUserName='filehelper')

    def worker_thread(self, conn, addr):
        conn.settimeout(args.timeout)
        if conn is not None:
            self.send('job connected! [{}]'.format(addr))
        else:
            self.send('none connection!')
            return -1

        while(True):
            try:
                recv_data = conn.recv(1024)
            except socket.timeout as e:
                print('no msg...')
            else:
                msg_len = len(recv_data)
                if msg_len == 0:
                    self.send('connection break, waiting for other connections..')
                    break
                self.send(str(recv_data, encoding = 'utf-8'))
        conn.close()

    def start_server(self, server_ip, server_port):
        ip_port = (server_ip, server_port)
        s = socket.socket()
        s.bind(ip_port)

        # dump ip/port info to file
        with open('server.txt', 'w') as f:
            f.write('{} {}\n'.format(server_ip, server_port))

        s.listen()
        self.send('server listening on {}, waiting for job connection...'.format(server_ip))
        while(True):
            conn, addr = s.accept()
            threading.Thread(target=self.worker_thread, args=(conn, addr)).start()

class MsgClient(object):
    def __init__(self, server_ip, server_port):
        self._init_client(server_ip, server_port)
        self.send('I\'m client!\n')

    def send(self, msg, echo=True):
        self.s.send(bytes(msg, encoding = 'utf-8'))
        if echo:
            print(msg)

    def _init_client(self, server_ip, server_port):
        ip_port = (server_ip, server_port)
        self.s = socket.socket()
        self.s.connect(ip_port)

    def close(self):
        self.s.close()

def itchat_manager():
    def run_and_get(cmd, screen=False):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        output = ''

        for line in process.stdout:
            line = line.decode('utf-8')
            if screen:
                print(line, end='', flush=True)
            output += line.strip(' ')

        return output

    @itchat.msg_register(itchat.content.TEXT)
    def text_reply(msg):
        res_txt = None
        cmd_dict = {
            'sq' : 'squeue -p VI_Face_V100',
            'sq1': 'squeue -p VI_Face_1080TI'
        }
        if msg.text in cmd_dict:
            cmd = cmd_dict[msg.text]
            res_txt = run_and_get(cmd)
        elif msg.text.startswith('exec:'):
            cmd = msg.text.replace('exec:', '')
            if os.system(cmd) == 0:
                res_txt = 'exec successed!'
            else:
                res_txt = 'exec failed!'
        elif msg.text.startswith('getinfo:'):
            cmd = msg.text.replace('getinfo:', '')
            res_txt = run_and_get(cmd)
        
        if res_txt is not None:
            itchat.send(res_txt, toUserName='filehelper')

    itchat.auto_login(enableCmdQR=2, hotReload=True)
    itchat.run(True)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'server':
        if args.ip is None or args.port is None:
            with open('server.txt', 'r') as f:
                line = f.read().strip().split()
                args.ip = line[0]
                args.port = int(line[1])
                print('reading ip & port from server.txt, {}:{}'.format(args.ip, args.port))
        s = MsgServer(args.ip, args.port)
    elif args.mode == 'manager':
        itchat_manager()
    else:
        s = MsgClient(args.ip, args.port)
        time.sleep(5)
        s.close()
