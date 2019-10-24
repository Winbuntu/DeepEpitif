

with open ("Ture_target_with_labels_1024.bed") as a:
    for line in a:
        elements = line.strip().split("\t")
        if elements[0] == "CHR":
            continue
        #print(   int(elements[2]) -   int(elements[1]) )

        if (  int(elements[2]) -   int(elements[1]) != 1024  ):
            print(line)