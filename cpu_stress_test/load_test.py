import multiprocessing
import time
import random
import os
from datetime import timedelta

def burn_cpu(stop_event):
    pid = os.getpid()

    try:
        while not stop_event.is_set():
            work_percent = random.uniform(0.80, 1.00)
            sleep_percent = 1.0 - work_percent
            
            cycle_duration_ms = 100 
            work_time_sec = (cycle_duration_ms / 1000.0) * work_percent
            sleep_time_sec = (cycle_duration_ms / 1000.0) * sleep_percent

            work_until = time.time() + work_time_sec
            while time.time() < work_until:
                if stop_event.is_set():
                    break
                _ = 10 * 10 
            
            if not stop_event.is_set():
                time.sleep(sleep_time_sec)

    except KeyboardInterrupt:
        pass
    finally:
        pass

def main():
    stress_duration_sec = 5 * 60
    min_wait_min = 1
    max_wait_min = 15
    num_cores = multiprocessing.cpu_count()
    
    print("=" * 60)
    print(f"Continuous Variable CPU Stress Test")
    print(f"Detected {num_cores} CPU cores.")
    print(f"Stress Duration: {stress_duration_sec / 60:.0f} minutes")
    print(f"Wait Interval: {min_wait_min} to {max_wait_min} minutes")
    print(f"Load Target: 80% - 100% (variable)")
    print("=" * 60)
    print("Press Ctrl+C to stop the test at any time.")

    pool = []
    stop_event = None

    try:
        while True:
            print(f"\n({time.strftime('%H:%M:%S')}) Starting 5-minute stress test on {num_cores} cores...")
            
            stop_event = multiprocessing.Event()
            pool = []

            for i in range(num_cores):
                p = multiprocessing.Process(target=burn_cpu, args=(stop_event,))
                p.start()
                pool.append(p)

            time.sleep(stress_duration_sec)

            print(f"({time.strftime('%H:%M:%S')}) Stopping worker processes...")
            stop_event.set()
            for p in pool:
                p.join()
            print("Stress test complete.")

            wait_duration_sec = random.uniform(min_wait_min * 60, max_wait_min * 60)
            wait_until = time.time() + wait_duration_sec
            
            print(f"({time.strftime('%H:%M:%S')}) Waiting for next run...")
            print(f"   (Next test will start in {str(timedelta(seconds=int(wait_duration_sec)))} at {time.strftime('%H:%M:%S', time.localtime(wait_until))})")
            
            while time.time() < wait_until:
                time.sleep(1)


    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected! Stopping all tasks...")
    
    finally:
        print("Shutting down...")
        
        if stop_event and not stop_event.is_set():
            print("   Signaling processes to stop...")
            stop_event.set()

        for p in pool:
            if p.is_alive():
                p.join()
        
        print("Test successfully terminated.")
        print("=" * 60)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()