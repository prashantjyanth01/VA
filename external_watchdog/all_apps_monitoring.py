# from Utils.context.context import ReadConfig
# from subprocess import Popen
# from Utils.logger.logger import LoggerClass
# import time
# import socket
#
#
# class OneWatch:
#     def __init__(self):
#         ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
#         ctx2 = ReadConfig("configs/LOCAL_CONFIG.xml")
#
#         self.localcontext = ctx2.context
#         self.watchdog_sleep = max(5, int(self.localcontext.get("EXTERNAL_WATCHDOG", {}).get("SLEEP", 5)))
#
#         self.context = ctx1.context
#         self.portsMap = {9997: "python3 basemanager.py",
#                          9977: "python3 videoagent.py",
#                          9998: "python3 xmlagent.py",
#                          9999: "python3 queuemanager_client.py"}
#         self.logging_obj = LoggerClass(m_type="EXTERNAL_WATCHDOG")
#         src_tree = self.context.get("ACTIVE_SOURCES_GROUPS", None)
#         if src_tree:
#             for src in src_tree['SRC_GRP']:
#                 if src['STATUS'] == "TRUE":
#                     src_id = src['ID']
#                     port_id = 9900 + int(src_id)
#                     self.portsMap.update({port_id: "python3 ees_app.py " + src_id})
#
#     def check_if_open(self, location):
#         isOpen = 0
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         try:
#             s.bind(location)
#         except Exception as e:
#             isOpen = 1
#             pass
#         s.close()
#         return isOpen
#
#     def start(self):
#         #Popen("python3 vms_api.py", shell=True)
#         while True:
#             for port, app in self.portsMap.items():
#                 location = ("127.0.0.1", port)
#                 if app == "python3 vms_api.py":
#                     continue
#
#                 result_of_check = self.check_if_open(location)
#
#                 if result_of_check == 0:
#
#                     self.logging_obj.log_msg(msg=app + ' not running', level=0)
#                     Popen(app, shell=True)
#                     self.logging_obj.log_msg(msg=app + ' started', level=0)
#                     pass
#                 else:
#                     print(app, ' is running')
#                     self.logging_obj.log_msg(msg=app + ' running', level=0)
#
#             time.sleep(self.watchdog_sleep)
#             print('\n\n')
#         pass

from Utils.context.context import ReadConfig
from subprocess import Popen
from Utils.logger.logger import LoggerClass
import time
import socket
from contextlib import closing


class OneWatch:
    def __init__(self):
        ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
        ctx2 = ReadConfig("configs/LOCAL_CONFIG.xml")

        self.localcontext = ctx2.context
        self.watchdog_sleep = max(5, int(self.localcontext.get("EXTERNAL_WATCHDOG", {}).get("SLEEP", 5)))

        self.context = ctx1.context
        self.udp_atcs_port = int(self.context['ATCS_SIGNAL_RECEIVER']['PORT'])
        self.portsMap = {9977: "python3 videoagent.py",
                         9997: "python3 basemanager.py",
                         9998: "python3 xmlagent.py",
                         9999: "python3 queuemanager_client.py",
                         self.udp_atcs_port : "python3 UDPServer.py"}

        self.logging_obj = LoggerClass(m_type="EXTERNAL_WATCHDOG")
        src_tree = self.context.get("ACTIVE_SOURCES_GROUPS", None)
        if src_tree:
            for src in src_tree['SRC_GRP']:
                if src['STATUS'] == "TRUE":
                    src_id = src['ID']
                    port_id = 9900 + int(src_id)
                    self.portsMap.update({port_id: "python3 ees_app.py " + src_id})

    def check_if_open(self, location):
        """
        Description: Here we are using two condition one for tcp servers and one for UDP server
        in my testing i found if a port is used for socket_DGRAM then our socket.SOCK_STREAM
        can be able to bind it so for that i used else if any one used different port it is
        mandatory to changed it in if condition as well (if location[-1]==<port to be used>' as well
        :param location: It's a tuple of (ip, port)
        :return: if port is bind to a port then return 0 othewise 1
        """
        isOpen = 0
        if location[-1] != self.udp_atcs_port:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(location)
            except Exception as e:
                isOpen = 1
                pass
            s.close()
            return isOpen
        else:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.bind(location)
            except Exception as e:
                isOpen = 1
                pass
            s.close()
            return isOpen


    def start(self):
        #Popen("python3 vms_api.py", shell=True)
        while True:
            for port, app in self.portsMap.items():
                location = ("127.0.0.1", port)
                if app == "python3 vms_api.py":
                    continue
                result_of_check = self.check_if_open(location)
                if result_of_check == 0:
                    self.logging_obj.log_msg(msg=app + ' not running', level=0)
                    Popen(app, shell=True)
                    self.logging_obj.log_msg(msg=app + ' started', level=0)
                    pass
                else:
                    print(app, ' is running')
                    self.logging_obj.log_msg(msg=app + ' running', level=0)

            time.sleep(self.watchdog_sleep)
            print('\n\n')
        pass

if __name__=="__main__":
    obj = OneWatch()
    obj.start()
