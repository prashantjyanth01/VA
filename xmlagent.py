"""
    Purpose     :   The module is used to transfer files and analyse file using sftp processing module
    Device      :   GPU Nvidia 1050Ti 4GB RAM (2)
    Author      :   Shamsher Singh
    Last Edited :   13 Jul 2020
    Edited By   :   Isha Gupta
    File        :   xmlagent.py
"""

from Utils.sftp.Modules.Core_Modules.vids_sftp_org import VidsSftp
from Watchdog.watchdog import *
import socket

DEFINED_PORT = 9998


def main():
    """
    Start running agent.
    :return     : None
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', DEFINED_PORT))
        xmlagent_obj = VidsSftp()
        xmlagent_obj.run()
    except Exception as e:
        print(e)
        watchdog_obj.log_exception(event_type="COMMUNICATION_AGENT", exception=e)


if __name__ == "__main__":
    # Communication Agent application start log
    watchdog_obj.log_watchdog_start(event_type="COMMUNICATION_AGENT")
    # run main
    print('RUNNING XML AGENT')
    main()
