from Utils.context.context import ReadConfig
from subprocess import Popen
from Utils.logger.logger import LoggerClass
import time
import socket


class OneWatch:
    def __init__(self):
        ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
        ctx2 = ReadConfig("configs/LOCAL_CONFIG.xml")

        self.localcontext = ctx2.context
        self.watchdog_sleep = max(5, int(self.localcontext.get("EXTERNAL_WATCHDOG", {}).get("SLEEP", 5)))

        self.context = ctx1.context
        self.portsMap = {9998: "app_exes/xmlagent",
                         9997: "app_exes/basemanager",
                         9999: "app_exes/queuemanager_client"}
        self.logging_obj = LoggerClass(m_type="EXTERNAL_WATCHDOG")
        self.portsMap.update({9910: "app_exes/ees_app Bootup"})
        self.portsMap.update({5050: "app_exes/vms_api"})

    def check_if_open(self, location):
        isOpen = 0
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(location)
        except Exception as e:
            isOpen = 1
            pass
        s.close()
        return isOpen

    def start(self):
        Popen("app_exes/vms_api", shell=True)
        while True:
            for port, app in self.portsMap.items():
                location = ("127.0.0.1", port)

                if app == "app_exes/vms_api":
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
