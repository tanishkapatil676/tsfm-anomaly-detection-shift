import sys
import time

print("Python version:", sys.version)
t0 = time.time()
print("Importing chronos...")
try:
    import chronos
    print("Chronos imported successfully in", round(time.time() - t0, 2), "s")
    from chronos import BaseChronosPipeline
    print("BaseChronosPipeline imported successfully!")
except Exception as e:
    print("Import error:", type(e).__name__, e)
