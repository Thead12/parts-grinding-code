from icmplib import multiping
from subprocess import call

while True:
    hosts = multiping(['158.125.93.1', '158.125.93.202'], interval=5)
    for host in hosts:
        if host.is_alive:
         print(f'{host.address} is up!')
         #pingtest.printtest()
         call(["python", "Downloads\pingtest.py"]) #specify full path of file - note for linux, path separators are '/' not '\'
        else:
         print(f'{host.address} is down!')

