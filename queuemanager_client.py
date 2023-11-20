from Modules.Core_modules.BASEMANAGER_CLIENT import BaseManagerClient
from Watchdog.watchdog import *
import socket

a = []

DEFINED_PORT = 9999


def main():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', DEFINED_PORT))
        new_local_context = local_context.context
        queue_ctx = ReadConfig(filename="./configs/queue_manager.xml").context

        new_local_context.update(queue_ctx)

        obj = BaseManagerClient(local_context=local_context)
        obj.start_ocr()
        obj.start_atcs_queue_thread()
        obj.start_make_model_engine()
        obj.load_vehicle_color()
        obj.process_queue()
    except Exception as e:
        print(e)
        watchdog_obj.log_exception(event_type="QUEUE_MANAGER", exception=e)


if __name__ == "__main__":
    # QUEUE_MANAGER application start log
    watchdog_obj.log_watchdog_start(event_type="QUEUE_MANAGER")

    # run main
    main()
