from Watchdog.watchdog import *
import threading
import socket
import time
from multiprocessing.managers import BaseManager
import re
from threading import Event, Thread
import os
import signal

class ClientHandler:
    def __init__(self,list_of_src_ID):
        self.lock=threading.Lock()
        self.UDPEvent=Event()
        self.server=None
        # self.tevent=threading.Event()
        self.Total_Arm=list_of_src_ID
        self.IDDict={'01':'Green Left','02':'Green Straight','03':'Green Right','04':'Green Common','05': 'Amber Left','06':'Amber Straight','07':'Amber Right',
                     '08':'Amber Common','09':'Red Left','10':'Red Straight', '11':'Red Right','12':'Red Common'}
        self.ArmDict={'ARM':[],'LAMPTYPE':[],'LAMPSTATUS':[],'WAITDURATION': 0}
        self.isArmDictEmpty=True
        self.m = None
        self.udp_queue=[]
        self.isBasemanagerConnnected=self.connectservices(list_of_src_ID)

    def connectservices(self,list_of_src_ID):
        # for i in range(2):
        ip = '127.0.0.1'
        try:
            udp_queue_id=[]
            for src_id in list_of_src_ID:
                BaseManager.register('get_udp_queue_'+src_id)
                udp_queue_id.append('get_udp_queue_'+src_id)

            self.m = BaseManager(address=(ip, 9997), authkey=b'efkon@')
            self.m.connect()
            for i in range(len(udp_queue_id)):
                temp=getattr(self.m , udp_queue_id[i])
                self.udp_queue.append(temp())
            print(self.udp_queue)
        except Exception as e:
            print("Exception")
            watchdog_obj.log_exception(event_type="UDP Server Base Manager servive", exception=e)
            return False

        return True

    @staticmethod
    def checksum(input_data="AB", checksumVal=0):
        temp = int(ord(input_data[1]))
        for i in range(2, len(input_data)):
            if int(ord(input_data[i])) == 42:
                break
            temp = temp ^ int(ord(input_data[i]))
        check = str(hex(temp))[2:].upper()
        try:
            check_int = int(check)
        except Exception as e:
            check_int = int(check, 16)
        try:
            checksumVal_int = int(checksumVal)
        except Exception as e:
            checksumVal_int = int(checksumVal, 16)

        if check_int == checksumVal_int:
            return True
        else:
            return False

    @staticmethod
    def getIndex(datalist,value):
        return datalist.index(value)

    def extractID(self, lampid_list=None):
        if lampid_list:
            packetCode=lampid_list[1]
            if packetCode == 'M01':
                waittimeduration=lampid_list[-1]
                try:
                    no_of_lamp=int(lampid_list[3])
                except Exception as e:
                    no_of_lamp=int(lampid_list[3],16)

                index=self.getIndex(lampid_list,packetCode)
                StatusONLamps=lampid_list[index+3:-2]
                print('List of lights which are in on condition',StatusONLamps," : ",no_of_lamp)
                print('Duration : ',waittimeduration)
                self.preparedatadict(ON_LAMPS=StatusONLamps,WAIT_DURATION=waittimeduration)

            elif packetCode == 'M02':
                waittimeduration=lampid_list[-1]
                index_flash=self.getIndex(lampid_list,'F')
                index_on =self.getIndex(lampid_list,'G')
                index_packet_code=self.getIndex(lampid_list,packetCode)
                print('M02', index_flash, index_on,index_packet_code)
                StatusONLamps=lampid_list[index_on+2:index_flash]
                StatusflashLamps=lampid_list[index_flash+2:-2]
                print('List of ON Lights : ', StatusONLamps)
                print('List of Flash Lights : ', StatusflashLamps)
                print('Duration : ', waittimeduration)
                self.preparedatadict(ON_LAMPS=StatusONLamps, FLASHING_LAMPS=StatusflashLamps, WAIT_DURATION=waittimeduration)

            # TO-DO : prepare Dict
            elif packetCode =='M04':
                waittimeduration = lampid_list[-1]
                print('List of lights which are in on condition','All On')
                print('Duration : ', waittimeduration)
                self.preparedatadictM04(waittimeduration)
                pass

            # TO-DO : prepare Dict
            elif packetCode =='M05':
                waittimeduration = lampid_list[-1]
                print('List of lights which are in on condition', 'All On')
                print('Duration : ', waittimeduration)
                self.preparedatadictM05(waittimeduration)
                pass

            # TO-DO : prepare Dict
            elif packetCode =='M06':
                waittimeduration = lampid_list[-1]
                print('List of lights which are in on condition', 'All On')
                print('Duration : ', waittimeduration)
                self.preparedatadictM06(waittimeduration)
                pass

            # TO-DO : prepare Dict
            elif packetCode =='M07':
                waittimeduration = lampid_list[-1]
                print('List of lights which are in on condition', 'All On')
                print('Duration : ', waittimeduration)
                self.preparedatadictM07(waittimeduration)
                pass

            # TO-DO : prepare Dict
            elif packetCode =='M08':
                waittimeduration = lampid_list[-1]
                print('List of lights which are in on condition', 'All On')
                print('Duration : ', waittimeduration)
                self.preparedatadictM08(waittimeduration)
                pass

            else:
                print(packetCode)

    def InsertKeyValues(self,key,Listofid):
        if key == 'ON_LAMPS':
            status='ON'
            for id in Listofid:
                if self.IDDict.get(id[1:-1], 0):
                    self.ArmDict['ARM'].append('ARM'+str(id[0]))
                    self.ArmDict['LAMPTYPE'].append(self.IDDict.get(id[1:-1], 0))
                    self.ArmDict['LAMPSTATUS'].append(status)
        if key == 'FLASHING_LAMPS':
            status='FLASHING'
            for id in Listofid:
                if self.IDDict.get(id[1:-1], 0):
                    self.ArmDict['ARM'].append('ARM'+str(id[0]))
                    self.ArmDict['LAMPTYPE'].append(self.IDDict.get(id[1:-1], 0))
                    self.ArmDict['LAMPSTATUS'].append(status)

    def preparedatadict(self,**kwargs):
        # if not self.isArmDictEmpty:
        #     print('Cleaning...')
        print(kwargs)
        for key, val in kwargs.items():
            if key == 'WAIT_DURATION' and self.ArmDict['WAITDURATION']==0:
                self.ArmDict['WAITDURATION']=val
            else:
                self.InsertKeyValues(key=key, Listofid=val)

    def preparedatadictM04(self,waitduration=0):
        status='ON'
        if len(self.Total_Arm) > 0:
            for id in self.Total_Arm:
                self.ArmDict['ARM'].append('ARM'+id)
                self.ArmDict['LAMPTYPE'].append('Red Common')
                self.ArmDict['LAMPSTATUS'].append(status)
            self.ArmDict['WAITDURATION'] = waitduration

    def preparedatadictM05(self,waitduration):
        status='ON'
        if len(self.Total_Arm) > 0:
            for id in self.Total_Arm:
                self.ArmDict['ARM'].append('ARM' + id)
                self.ArmDict['LAMPTYPE'].append('Amber Common')
                self.ArmDict['LAMPSTATUS'].append(status)
            self.ArmDict['WAITDURATION'] = waitduration

    def preparedatadictM06(self,waitduration):
        status='FLASHING'
        if len(self.Total_Arm) > 0:
            for id in self.Total_Arm:
                self.ArmDict['ARM'].append('ARM' + id)
                self.ArmDict['LAMPTYPE'].append('Red Common')
                self.ArmDict['LAMPSTATUS'].append(status)
            self.ArmDict['WAITDURATION'] = waitduration

    def preparedatadictM07(self,waitduration):
        status='FLASHING'
        if len(self.Total_Arm) > 0:
            for id in self.Total_Arm:
                self.ArmDict['ARM'].append('ARM' + id)
                self.ArmDict['LAMPTYPE'].append('Amber Common')
                self.ArmDict['LAMPSTATUS'].append(status)
            self.ArmDict['WAITDURATION'] = waitduration

    def preparedatadictM08(self,waitduration):
        status='OFF'
        if len(self.Total_Arm) > 0:
            for id in self.Total_Arm:
                self.ArmDict['ARM'].append('ARM' + id)
                self.ArmDict['LAMPTYPE'].append('ALL')
                self.ArmDict['LAMPSTATUS'].append(status)
            self.ArmDict['WAITDURATION'] = waitduration

    def cleandict(self):
        '''
            Cleaning Dictionary after processing it to queue to
            avoid
        '''
        print('Cleaning...........')
        for key in self.ArmDict.keys():
            if isinstance(self.ArmDict[key],list) and len(self.ArmDict[key]) != 0:
                self.ArmDict[key].clear()
            if isinstance(self.ArmDict[key],str) and int(self.ArmDict[key]) !=0:
                self.ArmDict[key]=0


    def decodedata(self,data):
        data_ = re.split(r"[,*$!\n]\s*", data)
        data_len=len(data_)
        try:
            for i in range(data_len):
                if data_[i] == '':
                    data_.remove('')
        except Exception as e:
            print(e)

        print(data_)
        if self.checksum(data,data_[-2]):
            if self.isBasemanagerConnnected:
                try:
                    #TO-Do Add condition
                    self.extractID(data_)
                    for q in self.udp_queue:
                        q.put(self.ArmDict)
                        time.sleep(.5)

                except Exception as e:
                    print(e)
                    return False
                self.cleandict()
                print(self.ArmDict)
        else:
            for q in self.udp_queue:
                q.put('Checksum Error...')
                time.sleep(.5)
        return True


    def run_(self,port,timeout):
        '''
        Only use when want to run on parallel thread.
        :return: Nothing
        '''
        try:
            if self.isBasemanagerConnnected:
                print('<< BaseManager Connected Successfully >> Starting UDP SERVER...')
                
                self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
                self.server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                self.server.bind(('0.0.0.0', port))
                # self.server.settimeout(timeout)
                while True:
                    try:
                        data, addr = self.server.recvfrom(1024)
                        #self.UDPEvent.set()
                        print(type(data), len(data),data,"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",port)
                        data_=data.decode('UTF-8')
                        print(type(data_))
                        if self.decodedata(data_):
                            self.UDPEvent.set()
                            print("Noisy...")
                            continue
                        print("received message: {0}--{1}".format(addr, data))
                    except Exception as e:
                        print('PK',e)
                        continue

        except Exception as e:
            print('Base manager Error')

    # def connect_server(timeout):
    #     server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    #     server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    #     server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    #     '''
    #     If a server binds to localhost, it will only listen to connections from the current machine to localhost.
    #     If a server binds to the external IP address of the machine, it will only listen to connections made to that
    #     IP address; it won't see connections to localhost. If a server binds to '',
    #     which is equivalent to 0.0.0.0, it will listen to both.
    #     '''
    #     server.bind(('0.0.0.0', DefinedPort))
    #     server.settimeout(timeout)
    #     return server



if __name__ == "__main__":
    watchdog_obj.log_watchdog_start(event_type="UDP_server")
    list_of_src_ID = []
    ctx1 = ReadConfig("configs/ENFORCEMENT_CONFIG.xml")
    context = ctx1.context
    src_tree = context.get("ACTIVE_SOURCES_GROUPS", None)
    source_groups = 0
    temp=None
    DefinedPort = int(context['ATCS_SIGNAL_RECEIVER']['PORT'])
    timeout= int(context['ATCS_SIGNAL_RECEIVER']['TIMEOUT'])
    if src_tree:
        for src in src_tree['SRC_GRP']:
            if src['STATUS'] == "TRUE" and src['ATCS_STATUS'] == "TRUE":
                try:
                    # temp = extractARMID(CONFIGS=src['CAM_CONFIGS']['CONFIG'])
                    # if temp:
                    #     temp = set(temp)
                    # print("src id ", ls)
                    list_of_src_ID.append(src['ID'])
                    source_groups += 1
                except Exception as e:
                    print(e)
            # for val in temp:
            #     ls.append(val)
            # temp.clear()

        # ls = list(set(ls))
    # obj1 = ClientHandler(ls)

    obj1 = ClientHandler(list_of_src_ID)
    t1=Thread(target=obj1.run_, args=(DefinedPort,timeout,))
    t1.start()

    while True:
        flag=obj1.UDPEvent.wait(timeout)
        if flag:
            print('Flag is set to true')
            obj1.UDPEvent.clear()
            continue
        if obj1.server is not None:
            print('Event _trigger timeout')
            obj1.server.close()
        os.kill(os.getpid(), signal.SIGKILL)

    # if obj1.isBasemanagerConnnected:
    #     print('<< BaseManager  Connected Successfully >> Starting UDP_SERVER...')
    #     try:
    #         server=connect_server(timeout)
    #         while True:
    #             try:
    #                 # rec = select.select()
    #                 data, addr = server.recvfrom(1024)
    #                 print(type(data), len(data))
    #                 data_ = data.decode('UTF-8')
    #                 print(type(data_))
    #                 if obj1.decodedata(data_):
    #                     print("Noisy...")
    #                     continue
    #                 print("received message: {0}--{1}".format(addr, data))
    #             except Exception as e:
    #                 print(e)
    #                 watchdog_obj.log_exception(event_type='UDP SERVER', exception=e)
    #                 server.close()
    #                 time.sleep(.3)
    #                 server=connect_server(timeout)
    #
    #     except Exception as e:
    #         print(e)
    #         watchdog_obj.log_exception(event_type='UDP SERVER STOPPED',exception=e)
    #

