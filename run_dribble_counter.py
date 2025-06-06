import traceback
import sys

try:
    from dribble_counting import DribbleCounter
    
    print("Starting dribble counter...")
    counter = DribbleCounter()
    counter.run()
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc(file=sys.stdout)
    
    print("\nPress Enter to exit...")
    input()
