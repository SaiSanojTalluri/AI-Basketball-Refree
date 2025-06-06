import traceback
import sys

try:
    from double_dribble import DoubleDribbleDetector
    
    print("Starting double dribble detector...")
    detector = DoubleDribbleDetector()
    detector.run()
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc(file=sys.stdout)
    
    print("\nPress Enter to exit...")
    input()
