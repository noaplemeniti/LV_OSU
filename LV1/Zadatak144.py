song_dict = { }

def main():
    one_word_count = 0
    f = open("song.txt")
    for line in f:
        for word in line.split():
            if(word not in song_dict.keys()):
                song_dict[word] = 1
            else:
                song_dict[word] = song_dict.get(word) + 1
    for key in song_dict.keys():
        if(song_dict[key] == 2):
            one_word_count += 1
    print("One word count: ", one_word_count*2)
    print(song_dict)
    return



if __name__=="__main__":
    main()