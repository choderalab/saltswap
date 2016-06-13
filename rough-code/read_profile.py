import cProfile
import pstats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read output profile data from cProfile")
    parser.add_argument('-f','--filename',type=str,help="the filename of the cProfile output",default="profile0")
    parser.add_argument('-n','--number',type=int,help="the top number of longest running algorithms by total time, default=50",default=50)
    args = parser.parse_args()

p = pstats.Stats(args.filename)
p.strip_dirs()
#p.strip_dirs().sort_stats(-1).print_stats()
#p.sort_stats('cumulative')
p.sort_stats('cumulative').print_stats(args.number)