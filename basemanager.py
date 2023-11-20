# from Modules.Core_modules.base_manager import server_app
# from Watchdog.watchdog import *
#
# def main():
#     try:
#         obj = server_app()
#         obj.run_it()
#     except Exception as e:
#         print(e)
#         watchdog_obj.log_exception(event_type="Basemanager", exception=e)
#
#
# if __name__ == "__main__":
#     watchdog_obj.log_watchdog_start(event_type="Basemanager")
#     # run main
#     print('running server app')
#     main()

from Modules.Core_modules.base_manager import server_app
from Watchdog.watchdog import *


#
# def extractARMID(**kwargs):
#     id=[]
#     num_of_sources=0
#     config_list = kwargs['CONFIGS']
#     if type(config_list) is dict:
#         config_list = [config_list]
#     for cam_config in config_list:
#         if cam_config["STATUS"] != "TRUE":
#             print("Cam config path {} deactivated".format(cam_config["FILE_PATH"]))
#             continue
#         file_path = cam_config["FILE_PATH"]
#         # filename = os.path.basename(file_path) + str(num_of_sources)
#         if file_path.endswith(".xml"):
#             temp = ReadConfig(filename=file_path)
#             if temp.context['TRAFFIC_LAMP_REGIONS']['ATCS_SIGNAL']['STATUS']:
#                 id.append(temp.context['CAM_INFO']['ARM_ID'])
#             # if temp.context['CAM_INFO']
#             # print(config_dict[filename].context[''])
#     return id

def main():
    list_of_src_ID=[]
    ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
    context = ctx1.context
    src_tree = context.get("ACTIVE_SOURCES_GROUPS", None)
    source_groups = 0
    if src_tree:
        for src in src_tree['SRC_GRP']:
            if src['STATUS'] == "TRUE" and src['ATCS_STATUS']=="TRUE":
                try:
                    # temp=extractARMID(CONFIGS=src['CAM_CONFIGS']['CONFIG'])
                    # if temp:
                    #     temp=set(temp)
                    source_groups += 1
                    list_of_src_ID.append(src['ID'])
                except Exception as e:
                    print(e)

            # for val in temp:
            #     ls.append(val)
            # temp.clear()
    try:
        # ls = list(set(ls))
        # obj = server_app(ls)
        obj = server_app(list_of_src_ID)
        obj.run_it()
    except Exception as e:
        print(e)
        watchdog_obj.log_exception(event_type="Basemanager", exception=e)


if __name__ == "__main__":
    watchdog_obj.log_watchdog_start(event_type="Basemanager")
    # run main
    print('running server app')
    main()