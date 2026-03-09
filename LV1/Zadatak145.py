def count_exclamations_in_spam(f):
    exc_count = 0
    for sms in f:
        sms = sms.strip()
        if sms.startswith("spam") and sms.endswith("!"):
            exc_count += 1
    return exc_count

def func(f):
    for sms in f:
        if sms.startswith("ham"):
            sms_split = sms.split()
            if sms_split[1][0] == "G":
                print(sms)
            

def average_words(f):
    total_ham = 0
    ham_count = 0
    total_spam = 0
    spam_count = 0
    for sms in f:
        if sms.startswith("ham"):
            ham_count += 1
            for word in sms.split():
                total_ham += 1
        elif sms.startswith("spam"):
            spam_count+=1
            for word in sms.split():
                total_spam += 1
        else:
            print("Error in list")
            return
    return(round(total_spam/spam_count, 2), round(total_ham/ham_count, 2))
    

def main():
    with open("SMSSpamCollection.txt") as f:
        spam_avg, ham_avg = average_words(f)
        f.seek(0)
        exc = count_exclamations_in_spam(f)
        f.seek(0)
        func(f)
    print(spam_avg, ham_avg)
    print(exc)


if __name__=="__main__":
    main()