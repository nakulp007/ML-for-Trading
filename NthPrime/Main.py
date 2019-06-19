__author__ = 'Nakul'

def main():
    ''' Main Function'''
    findNthPrime(545)

def findNthPrime(n):
    numPrimesFound = 0

    #iterate over all possible primes
    i=1
    while(numPrimesFound < n):
        i += 1
        #number to divide with
        for x in range(2, i+1):
            if(i%x == 0 and i!= x):
                break
            if(i==x):
                numPrimesFound += 1
    print i


if __name__ == '__main__':
    main()
