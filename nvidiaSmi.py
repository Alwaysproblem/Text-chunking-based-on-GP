import os
import time 

while True:
    try:
        os.system('nvidia-smi')
        time.sleep(1)
        os.system('cls')
    except KeyboardInterrupt :
        print("Interrupt by user.")
        import sys
        sys.exit(0)