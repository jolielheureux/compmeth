import argparse
import numpy as np

def freefall_time(height, gravity):
    """Return time for object to fall given a height and gravity
   Arguments:
        height: Drop height
        gravity: Gravitational acceleration
    Returns:
        time: Time to fall
   
   """
    time = np.sqrt((2*height)/(gravity))
    return time 

def main():
    parser = argparse.ArgumentParser(description="Calculate free fall time for a ball dropped from a specific height.")
    parser.add_argument("--height", type=float, required=True, help= "Drop height (meters)")
    parser.add_argument("--gravity", type=float, default=9.81, help = "Gravitational acceleration (meters per second ^ squared) Default: Earth's gravity - g = 9.8 m/s^2 ")
    args = parser.parse_args()

    if args.height < 0:
        parser.error("Height must be positive.")

    if args.gravity <= 0:
        parser.error("Gravity must be positive.")
    
    
    time = freefall_time(args.height, args.gravity)
    print(f'Time to reach the ground: {time:.3f} seconds')

if __name__ == "__main__":
    main()
    