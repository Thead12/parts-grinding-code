from icmplib import multiping
from subprocess import call
import psutil
import time

while True:
    #Checks to see if grinding machine is online - its Raspberry Pi will have an IP address
    counting_is_running = False
    while counting_is_running is False:
        hosts = multiping(['158.125.93.1'], interval=5) #replace with counting machine Raspberry Pi's IP address
        for host in hosts:
            if host.is_alive:
                print(f'--- {host.address} is up! ---')
                #pingtest.printtest()
                call(["python", "Downloads\pingtest.py"]) #specify full path of counting script file - note for linux, path separators are '/' not '\'
                counting_is_running = True
            else:
                print(f'--- {host.address} is down! ---')
    
    #Checks to see if the counting script is running whilst grinding machine is online
    while counting_is_running is True:
        processes = [p.cmdline() for p in psutil.process_iter() if p.name().lower() in ['python.exe']] 
        print(processes)
        print("---------")
        if ['C:\\Program Files\\Python311\\python.exe', '.\\Downloads\\running.py'] not in processes: #replace with the entry for the counting script
            print("Counting script not currently running!")
            counting_is_running = False 
        else:
            time.sleep(2)
