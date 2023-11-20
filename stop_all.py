import os


def kill_all_apps():
    os.system('pkill -9 -f run_all')
    os.system('pkill -9 -f ees_app')
    os.system('pkill -9 -f queuemanager_client')
    os.system('pkill -9 -f basemanager')
    os.system('pkill -9 -f xmlagent')
    os.system('pkill -9 -f vms_api')
    os.system('pkill -9 -f videoagent')
    os.system('fuser -n tcp -k 5050')


if __name__ == "__main__":
    kill_all_apps()
