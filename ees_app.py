import traceback
from Utils.Scheduler.rlvdScheduler import Scheduler
from Watchdog.watchdog import *
import socket
from threading import Thread
from threading import Event


class LoadContext:
    """
    LoadContext class reads arm config xml file and local config xml file.
    """

    def __init__(self, **kwargs):
        """
        get_arm_context class constructor reads arm config xml file and local config xml file based passed arm id.
        :param kwargs:
        arm_id :    1
        """
        self.activeArmsContext = None

        local_config = ReadConfig(filename="configs/LOCAL_CONFIG.xml", is_local_context=True)
        self.config_dict = {'LOCAL_CONFIG': local_config}

        self.num_of_sources = 0

        config_list = kwargs['CONFIGS']
        incidence_codes = kwargs["INCIDENTS"]
        sftp_file_transfer = kwargs["SFTP_FILE_TRANSFER"]
        properties_file = kwargs['PROPERTIES']
        srcgp_type = kwargs['TYPE']

        properties_config = ReadConfig(filename=properties_file, is_local_context=True).context

        properties_config.update({"INCIDENTS": incidence_codes, "SFTP_FILE_TRANSFER": sftp_file_transfer})

        self.config_dict['LOCAL_CONFIG'].context.update(properties_config)
        self.config_dict['LOCAL_CONFIG'].context.update({"TYPE":srcgp_type})

        if type(config_list) is dict:
            config_list = [config_list]
        for cam_config in config_list:
            if cam_config["STATUS"] != "TRUE":
                print("Cam config path {} deactivated".format(cam_config["FILE_PATH"]))
                continue
            file_path = cam_config["FILE_PATH"]
            filename = os.path.basename(file_path) + str(self.num_of_sources)
            if file_path.endswith(".xml"):
                self.config_dict[filename] = ReadConfig(filename=file_path)
                self.num_of_sources += 1


def main():
    bootup = "b"
    try:
        arg_src_id = "1"#sys.argv[1]
        print("Running src id ", arg_src_id)
    except Exception as e:
        print('Error : ' + str(e))
        arg_src_id = '2'
        print("running default src id 1")

    time.sleep(1)
    try:
        defined_port = 9900 + int(arg_src_id)
        print("DEFINED_PORT", defined_port)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        s.bind(('localhost', defined_port))
        # event for waiting forever
        wait_forever = Event()

        ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
        # print(ctx1.context['ACTIVE_SOURCES_GROUPS']['SRC_GRP'])
        context = ctx1.context
        src_tree = context.get("ACTIVE_SOURCES_GROUPS", None)
        incidence_codes = context.get("INCIDENTS", None)
        sftp_file_transfer = context.get("SFTP_FILE_TRANSFER", None)

        source_groups = 0

        jobs = []
        if src_tree:
            for src in src_tree['SRC_GRP']:
                if src['STATUS'] == "TRUE":
                    # print(src['TYPE'])
                    src_id = src['ID']
                    if arg_src_id != src_id:
                        continue
                    # port_id = int(src['ID'])
                    try:
                        print("src id ", src_id)
                        loaded_ctx = LoadContext(CONFIGS=src['CAM_CONFIGS']['CONFIG'],
                                                 PROPERTIES=src['PROPERTIES']['FILE_PATH'],
                                                 INCIDENTS=incidence_codes,
                                                 SFTP_FILE_TRANSFER=sftp_file_transfer,
                                                 TYPE=src['TYPE'])
                        if loaded_ctx is not None and loaded_ctx.num_of_sources == 0:
                            print("No active cam found under src id ", src_id)
                            continue
                        print(loaded_ctx)
                        app = Scheduler(serialContext=loaded_ctx,
                                        bootup=bootup,
                                        arm_id=src_id)
                        jobs.append(Thread(target=app.run))
                        Thread(target=app.run).start()

                        source_groups += 1

                    except Exception as e:
                        print('Error : ' + str(e) + traceback.format_exc())
                        watchdog_obj.log_exception(event_type="EES_APPLICATION", exception=e, arm_id=src_id)

        if source_groups == 0:
            exit(1)
        wait_forever.wait()
    except Exception as e:
        print("Error : ", e)


if __name__ == "__main__":
    # run main
    print('running ees_app')
    main()
