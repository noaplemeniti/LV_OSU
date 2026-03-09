import math

numbers = []

def avg(list):
    total = 0
    for element in list:
        total+=element
    return total/len(list)

def finish():
    print("Sorted list: ", numbers.sort())
    print("Count: ", len(numbers))
    print("Avg, min, max: ", avg(numbers),",", min(numbers),",", max(numbers), math.sqrt(numbers[-1]))
    return

def main():
    while True:
            number = input()
            if(number=="Done"): 
                 finish()
                 return
            numbers.append(int(number))


        

if __name__=="__main__":
    main()