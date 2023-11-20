"""
    Purpose     :   The module starts the RLVD App.
    Device      :   GPU Nvidia 1050Ti 4GB RAM (2)
    Author      :   Kanika Malhotra/ Shamsher Singh
    Last Edited :   31 Oct 2019
    Edited By   :   Kanika
    File        :   all_Arm_start.py
"""

from external_watchdog.all_apps_monitoring import OneWatch

obj = OneWatch()
obj.start()
