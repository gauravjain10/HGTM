import time
import sys
from submain_file import *

def main():
    try:
        print("\Config path is ", sys.argv[1])
    except:
        print("Config file path not found")
        return 
    try:
        print("\n No. of Events  ",sys.argv[2:])
    except:
        print("No. of Events are missing")
        return

    config_file_path=sys.argv[1]
    t1 = time.perf_counter()

    results= [submain(config_file_path,int(sys.argv[2]))]
    for result in results:
        print(result)
    t2 = time.perf_counter()
    print(f'Finished in {t2-t1} seconds')
    
if __name__ == "__main__":
    main()
