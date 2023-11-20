"""
    Purpose     :   To regularly monitor if each configured process/event is running
    Device      :   GPU Nvidia 1050Ti 4GB RAM (2)
    Author      :   Isha Gupta, Kanika Malhotra
    Last Edited :   21 May 2020
    Edited By   :   Isha Gupta, Kanika Malhotra
    File        :   watchdog.py
"""

import threading
from Utils.Helpers.helping import *
from Utils.logger.logger import LoggerClass


class MyWatchdogClass:

    def __init__(self):

        # Local config values
        self.watchdog_context = None
        self.log_dir_path = None
        self.is_debug = False
        self.event_types = []

        # Loading the Local configuration
        self.load_context()

        # Flag to monitor individual process
        self.arm_app_recv_flag = {}  # dict to monitor if frame receive cycle is running for each active arm
        self.arm_app_proc_flag = {}  # dict to monitor if frame process cycle is running for each active arm
        self.arm_noframe_flag = {}  # dict to monitor if bus is not available for each active arm
        self.queue_mgr_flag = False  # flag to monitor if queuemanager is running
        self.basemanager_flag = False  # flag to monitor if basemamager is running

        self.comm_in_process_flag = False  # flag to monitor the processing of one single file
        self.comm_agent_ext_flag = False  # flag to monitor if communication agent is running

        # Thread event for communication agent internal methods
        self.communication_internal_event = threading.Event()

        # logging class object
        self.logging_obj = LoggerClass(m_type="APPLICATION_WATCHDOG")

    def load_context(self):
        """
        Function to load local configuration
        """

        # Getting the local context
        context = local_context.context

        context_val = find_keys(["WATCHDOG", "LOGS_PATH"], context)

        self.watchdog_context = context_val[0]
        self.watchdog_context['EVENT'] = convert_list_dict(self.watchdog_context['EVENT'], 'TYPE')
        self.is_debug = self.watchdog_context['DEBUG_STATUS'] == 'TRUE'
        self.log_dir_path = context_val[1]

        self.watchdog_context['STATUS'] = convert_type(self.watchdog_context['STATUS'], 'bool')

        for k, v in self.watchdog_context['EVENT'].items():
            v['STATUS'] = convert_type(v['STATUS'], 'bool')
            v['MONITORING_TIME'] = convert_type(v['MONITORING_TIME'], 'int')

    def validate_event(self, event_type='', arm_id=''):
        """
        Flag based watchdog monitoring

        Steps:
        1. For the given event type, sleep the thread for the configured monitoring time
        2. Once the thread wakes up, check if the process has set the respective flag to True
        3. If flag is True then the process is working and set the flag back to False for the next cycle
        4. If flag is False then the process has crashed and restart the process

        Params:
        :event_type: type of event to monitor
        :arm_id: used for arm application event, key for each flag dict
        """

        while True:

            # For the given event type, sleep the thread for the configured monitoring time
            time.sleep(self.watchdog_context['EVENT'][event_type]['MONITORING_TIME'])

            if self.is_debug:
                print("For event - ", event_type)

            if event_type == 'ARM_APPLICATION':

                if self.is_debug:
                    print("=====================")
                    print("sleep over")
                    print("arm_app_recv_flag - ", self.arm_app_recv_flag[arm_id])
                    print("arm_noframe_flag - ", self.arm_noframe_flag[arm_id])
                    print("arm_app_proc_flag - ", self.arm_app_proc_flag[arm_id])
                    print("=====================")

                # Arm application crashes either in these scenarios:
                # 1. Frame receive thread crash or 2. Frame are being received but Process buffer crash
                if not self.arm_app_recv_flag[arm_id] or (self.arm_app_recv_flag[arm_id] and
                                                          not self.arm_noframe_flag[arm_id] and
                                                          not self.arm_app_proc_flag[arm_id]):
                    self.log_watchdog_restart(event_type=event_type,
                                              app_type="Arm Application for arm " + arm_id + " ")
                    restart()

                else:
                    self.log_watchdog_success(event_type=event_type + "for arm " + arm_id + " ")
                    self.arm_app_recv_flag[arm_id] = False
                    self.arm_app_proc_flag[arm_id] = False
                    self.arm_noframe_flag[arm_id] = False

            elif event_type == "QUEUE_MANAGER":

                # if self.basemanager_flag:
                #     # this event is raised when the basemanager has crashed
                #     # restart both basemanager and queuemanager
                #
                #     self.log_watchdog_restart(event_type="Basemanager", app_type="Basemanager")
                #     self.basemanager_flag = False
                #
                #     # kill the current basemanager process and restart
                #     os.system('pkill -9 -f basemanager.py')
                #     os.system('python3 basemanager.py & >> /dev/null')
                #     self.queue_mgr_flag = False

                # if basemanager is working then check for Queuemanager
                if not self.queue_mgr_flag:
                    self.log_watchdog_restart(event_type=event_type, app_type="QUEUE MANAGER")
                    restart()
                else:
                    self.log_watchdog_success(event_type=event_type)
                    self.log_watchdog_success(event_type="Basemanager")
                    self.queue_mgr_flag = False

            elif event_type == "COMMUNICATION_AGENT":

                # first check if the entire process got completed in the given monitoring time
                # if yes then all is working fine and no need to restart
                # if no then set the comm_in_process_flag to false and sleep for 1 min
                # if the time taken to process one single file is more than 1 min then something crashed, hence restart
                to_restart = False
                if not self.comm_agent_ext_flag:
                    self.comm_in_process_flag = False
                    time.sleep(60)
                    if not self.comm_in_process_flag:
                        to_restart = True

                if to_restart:
                    self.log_watchdog_restart(event_type=event_type, app_type="Communication Agent")
                    restart()
                else:
                    self.log_watchdog_success(event_type=event_type)
                    self.comm_agent_ext_flag = False
                    self.comm_in_process_flag = False

    def monitor_event(self, event_type=''):
        """
        Event based watchdog monitoring

        Steps:
        1. An event waits to be set till the configured timeout/monitoring time
        2. If the event is set before the timeout then clear the event for the next cycle
        3. If the event is not set after the timeout then restart the process

        Params:
        :event_type: type of event to monitor
        """

        if self.is_debug:
            print("For event : ", event_type)

        if event_type == 'DOWNLOAD_FILES' or event_type == 'UPLOAD_FILE_2_SERVER' \
                or event_type == 'DELETE_REMOTE_FILE':

            self.communication_internal_event.wait(self.watchdog_context['EVENT'][event_type]['MONITORING_TIME'])
            if not self.communication_internal_event.is_set():
                self.log_watchdog_restart(event_type=event_type, app_type="Communication Agent")
                restart()
            self.communication_internal_event.clear()

    def start_watchdog(self, event_type="", arm_id="", is_event=False):
        """
        Start the watchdog monitoring. Creates a thread for each monitoring process

        Params:
        :event_type: type of event to monitor
        :arm_id: used for arm application event, key for each flag dict
        :is_event: is the watchdog monitoring for the given process, event based or flag based
        """

        # if watchdog monitoring configured to be enabled
        # and if watchdog monitoring configured to be enabled for the given event
        if self.watchdog_context['STATUS'] and self.watchdog_context['EVENT'][event_type]['STATUS']:

            if self.is_debug:
                print("=====================")
                print("WD started for - ", event_type)
                print("=====================")

            # if flag based
            if not is_event:
                t1 = threading.Thread(target=self.validate_event, kwargs=dict(event_type=event_type, arm_id=arm_id))
                t1.start()

            # if event based
            else:
                t1 = threading.Thread(target=self.monitor_event, kwargs=dict(event_type=event_type))
                t1.start()

    def cancel_watchdog_(self, event_type=''):
        """
        Used only for event based monitoring.
        This sets the event (called by the individual process/event when their processing is complete)

        Params:
        :event_type: type of event to monitor
        """

        if self.is_debug:
            print("cancelling WD for " + event_type)

        if event_type == 'DOWNLOAD_FILES' or event_type == 'UPLOAD_FILE_2_SERVER' \
                or event_type == 'DELETE_REMOTE_FILE':
            self.communication_internal_event.set()

    def log_watchdog_restart(self, event_type="", app_type=""):
        """
        Log restart

        Params:
        :event_type: type of event to monitor
        :app_type: exact name of the event application
        """

        if self.is_debug:
            print(event_type + " timed out!")
            if event_type == "Basemanager":
                print("Going to restart the " + app_type)
            else:
                print("Going to restart the " + app_type + " due to watchdog timeout of " +
                      str(self.watchdog_context['EVENT'][event_type]['MONITORING_TIME']) + " sec")

        msg = event_type + " timed out!\nRestarting " + app_type + "\n\n"
        self.__log__(msg)

    def log_watchdog_success(self, event_type=""):
        """
        Log working of the application

        Params:
        :event_type: type of event to monitor
        """

        if self.is_debug:
            print(event_type + " working!")
            msg = event_type + " working!\n"
            self.__log__(msg)

    def log_watchdog_start(self, event_type="", src_id=""):
        if src_id is not None and src_id != "":
            msg = event_type + " for source " + src_id + " started!\n"
        else:
            msg = event_type + " started!\n"

        self.__log__(msg)

    def log_exception(self, event_type="", exception="", src_id=""):
        if src_id is not None and src_id != "":
            msg =event_type + " for source " + src_id + "\tReceived exception:\t" + str(exception) + \
                  "\n "
        else:
            msg = event_type + "\tReceived exception:\t" + str(exception) + "\n"
        self.__log__(msg)

    def __log__(self, msg=""):
        try:
            self.logging_obj.log_msg(msg=msg, level=0)
        except Exception as e:
            print('Error in Watchdog')


# Creating a global instance of MyWatchdogClass
watchdog_obj = MyWatchdogClass()
